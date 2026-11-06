from PIL import Image
import glob

for idx, p in enumerate(glob.glob("DataSet/StainScan/dataset/Cooking Oil_cotton/*.JPG")[:10]):
    im = Image.open(p)
    im.resize((200, 200)).save(f"scratch/oil_{idx}.jpg")

for idx, p in enumerate(glob.glob("DataSet/StainScan/dataset/Mud_cotton/*.JPG")[:5]):
    im = Image.open(p)
    im.resize((200, 200)).save(f"scratch/mud_{idx}.jpg")

print("Saved thumbnails")
