import os
import time
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
DATASET_DIR = r"DataSet/StainScan/dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)  # Standard input resolution for MobileNetV2
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_PATH_PTH = "stain_model.pth"
MODEL_PATH_H5 = "stain_model.h5"

# Analyze dataset brightness distribution
def check_dataset_brightness():
    print("\n" + "="*60)
    print("DATASET LIGHTING DIAGNOSTICS (HSV VALUE DISTRIBUTION)")
    print("="*60)
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: Dataset directory {DATASET_DIR} not found.")
        return
        
    for category in os.listdir(DATASET_DIR):
        cat_path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(cat_path):
            continue
        brightness_vals = []
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            try:
                img = Image.open(img_path)
                hsv_img = img.resize((64, 64)).convert("HSV")
                v_chan = np.array(hsv_img)[:, :, 2]
                brightness_vals.append(v_chan.mean() / 255.0)
            except Exception:
                pass
        if brightness_vals:
            avg_brightness = np.mean(brightness_vals)
            std_brightness = np.std(brightness_vals)
            print(f"Category: {category:<20} | Avg Brightness: {avg_brightness:.4f} (Std: {std_brightness:.4f})")
    print("="*60 + "\n")

# Custom transform block to stack HSV Saturation & Value channels onto RGB channels
class AddHSVChannels(object):
    def __call__(self, img):
        # Convert PIL RGB image to tensor: shape (3, H, W)
        rgb_tensor = transforms.ToTensor()(img)
        
        # Convert PIL RGB image to HSV space
        hsv_img = img.convert("HSV")
        hsv_tensor = transforms.ToTensor()(hsv_img)
        
        # Extract Saturation (channel index 1) and Value (channel index 2)
        s_channel = hsv_tensor[1:2, :, :]
        v_channel = hsv_tensor[2:3, :, :]
        
        # Concatenate: output tensor will have shape (5, H, W)
        x_5ch = torch.cat([rgb_tensor, s_channel, v_channel], dim=0)
        return x_5ch

# Define Custom Dataset wrapper to allow separate training and validation transforms
class DatasetSplitWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Custom Transfer Learning wrapper class for MobileNetV2 with 5-channel input support
class MobileNetV2Stain(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV2Stain, self).__init__()
        # Load MobileNetV2 base
        if pretrained:
            self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.base_model = models.mobilenet_v2()
            
        # Freeze standard base layer parameters
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # Modify first conv layer base_model.features[0][0] to accept 5 input channels
        original_conv = self.base_model.features[0][0]
        
        new_conv = nn.Conv2d(
            in_channels=5,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None
        )
        
        # Initialize weight layers
        with torch.no_grad():
            # Copy ImageNet pre-trained weights for R, G, B channels
            new_conv.weight[:, :3, :, :] = original_conv.weight
            # Set Saturation and Value channels weights to zero initially (neutral start)
            new_conv.weight[:, 3:, :, :] = 0.0
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias
                
        self.base_model.features[0][0] = new_conv
        
        # Unfreeze the first conv layer so it can train on the new channels
        for param in self.base_model.features[0][0].parameters():
            param.requires_grad = True
            
        # Replace output classification head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def save_as_h5(state_dict, file_path):
    print(f"Exporting model weights to HDF5 format: {file_path}")
    with h5py.File(file_path, 'w') as f:
        for name, tensor in state_dict.items():
            array = tensor.cpu().numpy()
            f.create_dataset(name, data=array)
    print("HDF5 export complete.")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on device: {device}")

    # Check Dataset lighting distribution
    check_dataset_brightness()

    # Preprocessing pipelines
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.7, 1.3)),
        AddHSVChannels(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        AddHSVChannels(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.5, 0.5])
    ])

    # Load Full Dataset
    full_dataset = ImageFolder(root=DATASET_DIR)
    classes = full_dataset.classes
    print(f"Detected Stain Categories: {classes}")

    # Train / Validation Split (80% Train, 20% Val)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    print(f"Total dataset size: {total_size} images. Training: {train_size}, Validation: {val_size}")

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_data = DatasetSplitWrapper(train_subset, transform=train_transform)
    val_data = DatasetSplitWrapper(val_subset, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize MobileNetV2 with pre-trained base features modified for 5 channels
    model = MobileNetV2Stain(num_classes=len(classes), pretrained=True).to(device)
    
    # Configure Class Weights to penalize defaulting to Mud
    # Class mapping: 0 -> Ballpen Ink, 1 -> Cooking Oil, 2 -> Mud (alphabetical order)
    class_weights = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimize classifier head + modified first layer parameters
    params_to_optimize = list(model.base_model.classifier[1].parameters()) + list(model.base_model.features[0][0].parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)

    best_val_acc = 0.0

    print("Beginning training loop...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
              
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"--> New best validation accuracy: {best_val_acc:.2f}%. Saving weights to {MODEL_PATH_PTH}...")
            torch.save(model.state_dict(), MODEL_PATH_PTH)

    print("\nTraining completed successfully!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Reload best state dict to run evaluation and export
    model.load_state_dict(torch.load(MODEL_PATH_PTH, weights_only=True))
    model.eval()
    
    # Generate Classification Report and Confusion Matrix on validation set
    print("\nEvaluating best model on Validation Set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Print metrics to stdout log
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("="*60)
    target_names = ["Black Ballpen Ink", "Used Cooking Oil", "Mud"]
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("="*60)
    print("CONFUSION MATRIX (Validation Set)")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    print("                  Predicted")
    print("                  Ink   Oil   Mud")
    print(f"Actual   Ink      {cm[0][0]:<5} {cm[0][1]:<5} {cm[0][2]:<5}")
    print(f"         Oil      {cm[1][0]:<5} {cm[1][1]:<5} {cm[1][2]:<5}")
    print(f"         Mud      {cm[2][0]:<5} {cm[2][1]:<5} {cm[2][2]:<5}")
    print("="*60 + "\n")
    
    # Print class-specific accuracies specifically for Oil and Ink classes
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
        
    print("="*60)
    print("CLASS-SPECIFIC ACCURACY RESOLUTION")
    print("="*60)
    for i in [0, 1]:  # Index 0 is Ink, Index 1 is Oil
        class_acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"Validation Accuracy for '{target_names[i]}': {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    print("="*60 + "\n")
    
    # Save H5 weights
    save_as_h5(model.state_dict(), MODEL_PATH_H5)

if __name__ == "__main__":
    train_model()
