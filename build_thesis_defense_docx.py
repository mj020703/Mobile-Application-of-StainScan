import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_OUT = r"c:\Users\John Michael Dohinog\Desktop\Mobile Application of StainScan\THESIS_DEFENSE_SUMMARY.docx"

doc = docx.Document()

# Set standard 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Colors
COLOR_PRIMARY = RGBColor(31, 78, 121)     # Deep Navy
COLOR_SECONDARY = RGBColor(41, 128, 185)  # Medium Blue
COLOR_DARK = RGBColor(40, 40, 40)         # Charcoal text
COLOR_MUTED = RGBColor(100, 100, 100)     # Gray text
HEX_NAVY = "1F4E79"
HEX_LIGHT_BG = "F2F5F8"
HEX_CODE_BG = "F7F9FA"
HEX_BORDER = "BDC3C7"
HEX_CALLOUT_BORDER = "1F4E79"
HEX_CALLOUT_BG = "F0F4F8"
HEX_ZEBRA = "F9FAFB"

# Helper for cell shading
def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

# Helper for cell borders
def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    borders = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    for b_name, b_val in borders.items():
        if b_val:
            element = parse_xml(f'<w:{b_name} {nsdecls("w")} w:val="{b_val.get("val", "single")}" w:sz="{b_val.get("sz", "4")}" w:space="0" w:color="{b_val.get("color", "CCCCCC")}"/>')
            tcBorders.append(element)
        else:
            element = parse_xml(f'<w:{b_name} {nsdecls("w")} w:val="none"/>')
            tcBorders.append(element)
    tcPr.append(tcBorders)

# Helper for cell margins (padding)
def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

# Helper for table header formatting
def style_table_header(row, col_widths=None):
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, HEX_NAVY)
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        set_cell_borders(cell, 
            top={"color": HEX_NAVY, "sz": "6"}, 
            bottom={"color": HEX_NAVY, "sz": "12"}, 
            left={"color": "FFFFFF", "sz": "2"}, 
            right={"color": "FFFFFF", "sz": "2"})
        if col_widths and idx < len(col_widths):
            cell.width = Inches(col_widths[idx])
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
                r.font.name = "Calibri"
                r.font.size = Pt(10)

def style_table_row(row, is_zebra=False, col_widths=None):
    bg_color = HEX_ZEBRA if is_zebra else "FFFFFF"
    for idx, cell in enumerate(row.cells):
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
        set_cell_borders(cell, 
            top={"color": "E5E7EB", "sz": "4"}, 
            bottom={"color": "E5E7EB", "sz": "4"}, 
            left={"color": "E5E7EB", "sz": "4"}, 
            right={"color": "E5E7EB", "sz": "4"})
        if col_widths and idx < len(col_widths):
            cell.width = Inches(col_widths[idx])
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(9.5)
                r.font.color.rgb = COLOR_DARK

def add_callout_box(doc, title, paragraphs_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=160, bottom=160, left=220, right=200)
    set_cell_borders(cell, 
        left={"val": "single", "sz": "24", "color": HEX_CALLOUT_BORDER}, 
        top={"val": "single", "sz": "4", "color": "D1D5DB"}, 
        bottom={"val": "single", "sz": "4", "color": "D1D5DB"}, 
        right={"val": "single", "sz": "4", "color": "D1D5DB"})
    
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(2)
    p0.paragraph_format.space_after = Pt(4)
    r_title = p0.add_run(title)
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = COLOR_PRIMARY

    for text in paragraphs_text:
        p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_CODE_BG)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    set_cell_borders(cell, 
        left={"val": "single", "sz": "18", "color": "6B7280"}, 
        top={"val": "single", "sz": "4", "color": "E5E7EB"}, 
        bottom={"val": "single", "sz": "4", "color": "E5E7EB"}, 
        right={"val": "single", "sz": "4", "color": "E5E7EB"})
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(code_text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

# ---------------------------------------------------------
# COVER / HEADER
# ---------------------------------------------------------
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after = Pt(2)
run_title = p_title.add_run("StainScan: Thesis Defense Technical Cheat Sheet")
run_title.font.name = "Calibri"
run_title.font.size = Pt(22)
run_title.font.bold = True
run_title.font.color.rgb = COLOR_PRIMARY

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_before = Pt(0)
p_sub.paragraph_format.space_after = Pt(2)
run_sub = p_sub.add_run("System Architecture, Machine Learning Pipeline, Dataset Specifications & Defense Guide")
run_sub.font.name = "Calibri"
run_sub.font.size = Pt(12)
run_sub.font.bold = True
run_sub.font.color.rgb = COLOR_SECONDARY

p_inst = doc.add_paragraph()
p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_inst.paragraph_format.space_before = Pt(0)
p_inst.paragraph_format.space_after = Pt(14)
run_inst = p_inst.add_run("Academic Year 2025–2026 | STI West Negros University")
run_inst.font.name = "Calibri"
run_inst.font.size = Pt(10.5)
run_inst.font.italic = True
run_inst.font.color.rgb = COLOR_MUTED

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = COLOR_PRIMARY

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = COLOR_SECONDARY

def add_body(text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Calibri"
        rb.font.size = Pt(10)
        rb.font.bold = True
        rb.font.color.rgb = COLOR_DARK
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK
    return p

def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Calibri"
        rb.font.size = Pt(10)
        rb.font.bold = True
        rb.font.color.rgb = COLOR_DARK
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK
    return p

# =========================================================
# SECTION 1: SYSTEM & ARCHITECTURE OVERVIEW
# =========================================================
add_h1("1. System & Architecture Overview")

add_h2("1.1 Machine Learning Architecture & CNN Model Design")
add_body(
    "StainScan utilizes Transfer Learning based on the lightweight MobileNetV2 convolutional neural network "
    "backbone pre-trained on ImageNet. The architecture is engineered specifically for low-latency, high-accuracy "
    "classification of fabric stains on mobile devices and resource-constrained cloud containers."
)

arch_flow = (
    "Input Image (RGB)\n"
    "      │\n"
    "      ▼\n"
    "Bilinear Resize to (224, 224)\n"
    "      │\n"
    "      ▼\n"
    "CIELAB CLAHE Contrast Equalization (L* Channel, clipLimit=2.0, tileGrid=(8,8))\n"
    "      │\n"
    "      ▼\n"
    "MobileNetV2 preprocess_input (Pixel values normalized to [-1.0, 1.0])\n"
    "      │\n"
    "      ▼\n"
    "MobileNetV2 Feature Extractor (154 layers, Inverted Residuals, Depthwise Separable Convolutions)\n"
    "      │ [Output Shape: (None, 7, 7, 1280) | 2,257,984 Parameters]\n"
    "      ▼\n"
    "GlobalAveragePooling2D (Spatial compression to 1,280-dimensional feature vector)\n"
    "      │\n"
    "      ▼\n"
    "BatchNormalization (bn_head) (Stabilizes activation scales | 5,120 Parameters)\n"
    "      │\n"
    "      ▼\n"
    "Dropout (rate = 0.40) (Prevents co-adaptation on fabric weave patterns)\n"
    "      │\n"
    "      ▼\n"
    "Dense Classifier Head (units = 3, activation = 'softmax') | 3,843 Parameters\n"
    "      │\n"
    "      ▼\n"
    "Predicted Stain Probabilities: [Black Ballpen Ink, Used Cooking Oil, Mud]"
)
add_code_block(doc, arch_flow)

add_body("Detailed Layer-by-Layer Architecture Specifications:", bold_prefix="Table 1: ")

tbl_arch = doc.add_table(rows=7, cols=5)
tbl_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_arch = [1.2, 1.8, 1.2, 1.0, 1.3]

headers_arch = ["Layer / Component", "Type / Configuration", "Output Shape", "Parameters", "Purpose & Rationale"]
for idx, text in enumerate(headers_arch):
    tbl_arch.cell(0, idx).paragraphs[0].text = text
style_table_header(tbl_arch.rows[0], widths_arch)

data_arch = [
    ("Input Layer", "InputLayer", "(None, 224, 224, 3)", "0", "Standard high-resolution input tensor for MobileNetV2"),
    ("Base Backbone", "MobileNetV2 (ImageNet weights, include_top=False)", "(None, 7, 7, 1280)", "2,257,984", "Low-latency depthwise separable convolutions & bottleneck residuals"),
    ("Pooling Layer", "GlobalAveragePooling2D", "(None, 1280)", "0", "Averages spatial dimensions into 1D feature vector"),
    ("Normalization", "BatchNormalization (bn_head)", "(None, 1280)", "5,120", "Normalizes activation scales, accelerating head convergence"),
    ("Regularization", "Dropout (rate = 0.40)", "(None, 1280)", "0", "Prevents overfitting to specific cotton thread patterns"),
    ("Classifier Head", "Dense (activation = 'softmax')", "(None, 3)", "3,843", "Outputs normalized posterior probabilities across 3 stain classes")
]

for row_idx, data_row in enumerate(data_arch, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_arch.cell(row_idx, col_idx).paragraphs[0].text = val
    style_table_row(tbl_arch.rows[row_idx], is_zebra=(row_idx % 2 == 0), col_widths=widths_arch)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_bullet("Total Parameters: 2,266,947 (approx. 8.65 MB file footprint).", "• ")
add_bullet("Trainable Parameters: 1,201,283 (during Phase 2 fine-tuning).", "• ")
add_bullet("Non-Trainable Parameters: 1,065,664 (frozen lower backbone layers and batch statistics).", "• ")
add_bullet("Model Storage Artifacts: stainscan_model_v3_balanced.keras and stainscan_model_v3_balanced.h5.", "• ")

add_h2("Two-Phase Training Strategy")
add_bullet(
    "Phase 1: Feature Extraction (20 Epochs, lr = 1e-3, Adam, Categorical Crossentropy). "
    "The pre-trained MobileNetV2 base is completely frozen (base_model.trainable = False). "
    "Only the newly initialized custom classification head (GAP, BatchNormalization, Dropout, Dense) is trained.",
    "Phase 1: "
)
add_bullet(
    "Phase 2: Fine-Tuning (10 Epochs, lr = 1e-5, Adam, Categorical Crossentropy). "
    "The top 20 layers of MobileNetV2 are unfrozen (base_model.layers[:-20] frozen). "
    "All BatchNormalization layers across the entire base model remain explicitly frozen (layer.trainable = False) "
    "to preserve the running mean and variance statistics learned from ImageNet.",
    "Phase 2: "
)
add_bullet(
    "Learning Rate Schedulers: ReduceLROnPlateau monitors val_loss (factor=0.5, min_lr=1e-7). "
    "ModelCheckpoint continuously saves the checkpoint yielding the highest validation accuracy.",
    "Callbacks: "
)

add_h2("1.2 Tech Stack Across System Layers")

tbl_tech = doc.add_table(rows=14, cols=4)
tbl_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_tech = [1.5, 1.6, 1.0, 2.4]

headers_tech = ["Layer / Role", "Technology / Library", "Version", "Specific Purpose in StainScan"]
for idx, text in enumerate(headers_tech):
    tbl_tech.cell(0, idx).paragraphs[0].text = text
style_table_header(tbl_tech.rows[0], widths_tech)

data_tech = [
    ("Backend Framework", "Flask", "3.0.3", "REST API routing (/predict, /api/scans, /api/metrics, /api/kb)"),
    ("API Middleware", "Flask-CORS", "4.0.1", "Cross-Origin Resource Sharing for hybrid APK & admin portal"),
    ("Production WSGI", "Gunicorn", "22.0.0", "Thread-pooled HTTP server (1 worker, 4 threads, 120s timeout)"),
    ("Deep Learning Engine", "TensorFlow / Keras", "2.16.1 / 3.3.0", "Model definition, transfer learning, weights serialization, inference"),
    ("Computer Vision", "OpenCV (cv2)", "4.9.0+", "CIELAB color transformation & CLAHE adaptive histogram equalization"),
    ("Image Processing", "Pillow (PIL)", "10.3.0", "Image decoding, bilinear resampling, central luminance detection"),
    ("RAW Image Decoding", "rawpy", "0.21.0+", "Sensor extraction & demosaicing of Canon DSLR .CR2 grey fabric photos"),
    ("Evaluation & Splitting", "scikit-learn", "1.4.2", "Stratified 80/20 train/val split, confusion matrix, precision/recall"),
    ("Numerical Math", "NumPy", "1.26.4", "Tensor array operations & compressed .npz dataset cache loading"),
    ("Database Driver", "PyMongo / dnspython", "4.7.2 / 2.6.1", "MongoDB Atlas cloud cluster communication via TLS & DNS SRV"),
    ("Security Certificates", "certifi", "2024.7.4", "Mozilla root CA certificate bundle for secure TLS Atlas handshakes"),
    ("Frontend Languages", "HTML5, Vanilla CSS3, JS", "ES6+", "Native responsive UI, glassmorphic styling, canvas image handlers"),
    ("Mobile Packaging", "Capacitor / WebIntoApp", "v5+ / Cloud", "Native Android APK packaging (com.stainscan.app) via WebView bridge")
]

for row_idx, data_row in enumerate(data_tech, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_tech.cell(row_idx, col_idx).paragraphs[0].text = val
    style_table_row(tbl_tech.rows[row_idx], is_zebra=(row_idx % 2 == 0), col_widths=widths_tech)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_h2("1.3 Mobile App Architecture")
add_bullet("Hybrid Mobile Wrapper: Packaged into an installable Android APK (com.stainscan.app) via modern WebView bridge technologies (Capacitor / WebIntoApp).", "• ")
add_bullet("Dynamic Host Resolution: app.js sniffs the runtime environment. When running inside the packaged APK (file:// protocol), requests automatically route to the cloud API at https://stainscan-backend-gmbw.onrender.com. When running locally, it targets localhost:5000.", "• ")
add_bullet("Camera Capture: Uses HTML5 Media Capture (<input type='file' accept='image/*' capture='environment'>) to access device cameras, streaming frames into a Canvas element for Base64 encoding.", "• ")
add_bullet("Bilingual Support: Full internationalization supporting both English and Hiligaynon (Ilonggo), catering to local users in Western Visayas.", "• ")
add_bullet("Offline Resilience: LocalStorage caches scan records, user credentials, and cleaning recipes, preventing interface crashes during connectivity drops.", "• ")

add_h2("1.4 Data Storage & Database Configuration")
add_bullet("Database Service: MongoDB Atlas (M0 Cloud Cluster, database: stainscan_db).", "• ")
add_bullet("Connection Configuration: PyMongo connects using tls=True, tlsAllowInvalidCertificates=True, and serverSelectionTimeoutMS=5000. Certificate authority validation is handled by certifi.", "• ")
add_bullet("Graceful Failover: If MONGO_URI is missing or Atlas is unreachable, the backend seamlessly switches to Demo Mode (is_demo_mode = True), persisting records locally in scans.json and metrics.json.", "• ")
add_bullet("Collections: 'users' (accounts & roles), 'scans' (classification history & treatment status), 'kb' (cleaning steps & cautions), 'metrics' (aggregate counters), and 'logs' (system audit logs).", "• ")

# =========================================================
# SECTION 2: DATASET & PREPROCESSING PIPELINE
# =========================================================
add_h1("2. Dataset & Preprocessing Pipeline")

add_h2("2.1 Exact Dataset Distribution & Counts")
add_body(
    "The dataset consists of 1,800 physical photographs captured under realistic conditions on 100% cotton fabric. "
    "The dataset achieves an exact 1:1 balance between white cotton and grey cotton substrates."
)

tbl_data = doc.add_table(rows=5, cols=6)
tbl_data.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_data = [1.8, 1.2, 1.1, 1.1, 0.9, 0.9]

headers_data = ["Class Folder", "Stain Display Label", "White Cotton (.JPG)", "Grey Cotton (.CR2)", "Total", "Share"]
for idx, text in enumerate(headers_data):
    tbl_data.cell(0, idx).paragraphs[0].text = text
style_table_header(tbl_data.rows[0], widths_data)

data_dataset = [
    ("Ballpen Ink_cotton", "Black Ballpen Ink", "300", "300", "600", "33.33%"),
    ("Cooking Oil_cotton", "Used Cooking Oil", "300", "300", "600", "33.33%"),
    ("Mud_cotton", "Mud", "300", "300", "600", "33.33%"),
    ("TOTAL (Physical)", "All 3 Classes", "900 (50.0%)", "900 (50.0%)", "1,800", "100.0%")
]

for row_idx, data_row in enumerate(data_dataset, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_data.cell(row_idx, col_idx).paragraphs[0].text = val
    style_table_row(tbl_data.rows[row_idx], is_zebra=(row_idx == 4), col_widths=widths_data)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_bullet("Stratified 80/20 Partitioning: 1,440 training images (480 per class) and 360 validation images (120 per class).", "• ")
add_bullet("Substrate Parity: Exactly 50% white cotton (240 train / 60 val) and 50% grey cotton (240 train / 60 val) per class.", "• ")
add_bullet("Compressed Cache: Saved to dataset_clahe_224.npz containing 1,800 uint8 arrays of shape (224, 224, 3).", "• ")

add_h2("2.2 Contrast-Invariant Preprocessing & Data Augmentation")
add_body(
    "The Challenge: White cotton reflects high luminance, creating sharp contrast with stains. "
    "Grey cotton absorbs significantly more light; dark used cooking oil and mud exhibit low luminance differentials "
    "against dark grey fabric, confusing standard RGB classifiers."
)
add_bullet(
    "CIELAB Color Decoupling: RGB images are converted to CIELAB (cv2.COLOR_RGB2LAB), separating lightness (L*, 0-100) "
    "from chrominance opponent channels (a* green-red, b* blue-yellow).",
    "1. Color Space Separation: "
)
add_bullet(
    "CLAHE on L* Channel: Contrast Limited Adaptive Histogram Equalization (clipLimit=2.0, tileGridSize=(8,8)) is applied "
    "exclusively to the L* channel. This equalizes localized contrast around stain edges on grey fabric without amplifying background sensor noise.",
    "2. Adaptive Contrast Limiting: "
)
add_bullet(
    "Chrominance Integrity: Recombining equalized L* with unaltered a* and b* channels preserves the true spectral hues of stains "
    "(yellowish lipid tinge, blue-black ink tones, earthy soil tones).",
    "3. Color Fidelity: "
)
add_bullet(
    "Train-Inference Parity: The exact apply_clahe() function is executed in both train.py and server.py, eliminating train-serve skew.",
    "4. Preprocessing Match: "
)
add_bullet(
    "Data Augmentation (Train set only): Random horizontal/vertical flips, brightness shifts (max_delta=25), contrast scaling (0.7-1.4), "
    "and HSV saturation/hue jitter (saturation 0.8-1.2, hue delta 0.05).",
    "5. Robust Augmentation: "
)

add_h2("2.3 Model Validation Performance Metrics")
add_body("Evaluation on 360 unseen test images (120 per class):", bold_prefix="Final Validation Results: ")

tbl_metrics = doc.add_table(rows=5, cols=5)
tbl_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_metrics = [1.8, 1.2, 1.2, 1.2, 1.1]

headers_metrics = ["Target Stain Class", "Precision", "Recall", "F1-Score", "Test Samples"]
for idx, text in enumerate(headers_metrics):
    tbl_metrics.cell(0, idx).paragraphs[0].text = text
style_table_header(tbl_metrics.rows[0], widths_metrics)

data_metrics = [
    ("Black Ballpen Ink", "0.9917", "0.9917", "0.9917", "120"),
    ("Used Cooking Oil", "1.0000", "0.9917", "0.9958", "120"),
    ("Mud", "0.9917", "1.0000", "0.9959", "120"),
    ("Overall Model Accuracy", "0.9945 (Macro)", "0.9944 (Macro)", "0.9944 (Macro)", "360 (99.44%)")
]

for row_idx, data_row in enumerate(data_metrics, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_metrics.cell(row_idx, col_idx).paragraphs[0].text = val
    style_table_row(tbl_metrics.rows[row_idx], is_zebra=(row_idx == 4), col_widths=widths_metrics)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

cm_text = (
    "Confusion Matrix (Validation Set - 360 Images):\n"
    "                     PREDICTED\n"
    "                 Ink    Oil    Mud\n"
    "ACTUAL  Ink   [  119      0      1  ]  ──> 99.17% (119/120 Correct)\n"
    "        Oil   [    1    119      0  ]  ──> 99.17% (119/120 Correct)\n"
    "        Mud   [    0      0    120  ]  ──> 100.0% (120/120 Correct)\n\n"
    "Overall Validation Accuracy: 99.44% (358 / 360 Correct Predictions)"
)
add_code_block(doc, cm_text)

# =========================================================
# SECTION 3: DEPLOYMENT & HOSTING INFRASTRUCTURE
# =========================================================
add_h1("3. Deployment & Hosting Infrastructure")

add_h2("3.1 Backend Hosting Architecture & Render Configuration")
add_bullet("Host Platform: Render Cloud Web Service (https://stainscan-backend-gmbw.onrender.com).", "• ")
add_bullet("Container Image: python:3.10-slim with build-essential and libgomp1 (OpenMP CPU acceleration).", "• ")
add_bullet("WSGI Server: Gunicorn running 'gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 server:app'.", "• ")
add_bullet("Memory Management: Using 1 worker with 4 threads avoids duplicating TensorFlow weights across processes, keeping RAM usage below Render's 512 MB ceiling.", "• ")
add_bullet("Cold Start Pre-Warming: load_model() executes at startup and runs a dummy tensor through the graph, eliminating initial compilation delay.", "• ")

add_h2("3.2 Primary REST API Endpoints")

tbl_api = doc.add_table(rows=8, cols=4)
tbl_api.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_api = [1.2, 1.8, 1.5, 2.0]

headers_api = ["HTTP Method", "Endpoint Route", "Payload / Parameters", "Functional Description"]
for idx, text in enumerate(headers_api):
    tbl_api.cell(0, idx).paragraphs[0].text = text
style_table_header(tbl_api.rows[0], widths_api)

data_api = [
    ("POST", "/predict", "{ image: base64, fabric_color: '...' }", "Primary CNN inference, fabric color detection, returns cleaning steps"),
    ("GET", "/health", "None", "Health probe reporting model status and memory state"),
    ("GET", "/api/model-info", "None", "Returns input shapes, model file size, and class mappings"),
    ("POST", "/api/login", "{ email, password }", "Authenticates mobile users and system administrators"),
    ("GET / POST", "/api/scans", "{ id, stain, fabric, confidence }", "Fetches user history or logs a completed scan event"),
    ("PUT", "/api/scans/<id>", "{ status: 'Treated' }", "Updates laundry scan treatment state"),
    ("POST", "/api/treatment-complete", "{ scan_id: '...' }", "Increments treatment metrics with duplicate prevention")
]

for row_idx, data_row in enumerate(data_api, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_api.cell(row_idx, col_idx).paragraphs[0].text = val
    style_table_row(tbl_api.rows[row_idx], is_zebra=(row_idx % 2 == 0), col_widths=widths_api)

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_h2("3.3 Dynamic Fabric-Aware Stain Recommendation Engine")
add_body(
    "StainScan does not simply output a label; it connects the classification to a rule-based laundry protocol "
    "customized according to the fabric shade (detected dynamically via central crop luminance):"
)
add_bullet(
    "White Cotton: 70% Isopropyl alcohol blot -> liquid detergent pre-treat -> machine wash with oxygen bleach. "
    "Grey Cotton: Reverse-side alcohol blot -> color-safe detergent -> cool wash. Strictly NO chlorine bleach or acetone to prevent dye stripping.",
    "Black Ballpen Ink: "
)
add_bullet(
    "White Cotton: Baking soda absorption (15m) -> liquid dish soap scrub -> hot water rinse. "
    "Grey Cotton: Mild dish soap -> soft-bristled brush -> lukewarm water. Avoid aggressive friction to prevent surface pilling.",
    "Used Cooking Oil: "
)
add_bullet(
    "White Cotton: Dry mud fully -> scrape crust -> warm water pre-soak with oxygen bleach -> warm machine wash. "
    "Grey Cotton: Dry mud fully -> soft brush crust removal -> cool water pre-soak with color-safe detergent. Cool water avoids fixing clay into colored cotton.",
    "Mud: "
)

# =========================================================
# SECTION 4: PANEL DEFENSE FAQ & ANSWERS
# =========================================================
add_h1("4. Panel Defense FAQ & Answers")

add_callout_box(doc, "Question 1: Why did you choose MobileNetV2 over deeper architectures like ResNet-50 or custom shallow CNNs?", [
    "Defense Answer: We selected MobileNetV2 because of its optimal balance of representation capacity, parameter efficiency, and low inference latency on resource-constrained platforms.",
    "Architectures like ResNet-50 contain over 25.6 million parameters and demand high memory bandwidth, which would easily breach the 512 MB RAM limit of our free-tier container and introduce 800ms+ latency on mobile clients. Conversely, training a shallow CNN from scratch on 1,800 images risks overfitting and lacks the generalized edge filters learned from massive datasets.",
    "MobileNetV2 uses Inverted Residuals with Depthwise Separable Convolutions, factorizing standard 3x3 convolutions into a 3x3 depthwise spatial filter and a 1x1 pointwise projection. This reduces parameters to only ~2.27M and computational load to ~300M FLOPs. Initializing with ImageNet weights transferred rich low-level contour and texture primitives, allowing us to reach 99.44% validation accuracy with sub-150ms inference on CPU."
])

add_callout_box(doc, "Question 2: Stains on grey cotton have significantly lower visual contrast than on white cotton. How does your pipeline prevent the model from confusing fabric color with stain identity?", [
    "Defense Answer: We resolved the substrate contrast problem using a dual strategy combining physical dataset balancing and color-space-aware preprocessing:",
    "1. Dataset Level: We enforced a strict 1:1 physical balance: exactly 900 white cotton photos and 900 grey cotton photos, uniformly distributed across all classes (300:300 per class). This mathematically prevented the network's loss gradient from biasing toward white-fabric features.",
    "2. Preprocessing Level: Standard RGB histogram equalization distorts stain colors because RGB channels are correlated. We designed a CIELAB CLAHE pipeline: we convert RGB to CIELAB, isolating the lightness channel (L*) from chromatic opponent channels (a* and b*). We apply CLAHE (clipLimit=2.0, 8x8 tiles) solely to L*. This amplifies localized edge contrast along stain boundaries on grey fabric while leaving the color coordinates untouched. The model thus recognizes invariant spectral and textural features rather than background brightness."
])

add_callout_box(doc, "Question 3: How does your system distinguish dark used cooking oil from mud on dark grey cotton, given their visual similarity?", [
    "Defense Answer: Although used cooking oil and mud can both appear as dark blemishes on grey fabric to the naked eye, their underlying physical and optical properties create distinct feature maps in a convolutional neural network:",
    "1. Texture and Specular Reflectance: Cooking oil is a viscous lipid that saturates cotton fibers, creating a semi-translucent sheen that alters the index of refraction and produces smooth specular highlights along yarn ridges. Mud, by contrast, is a suspension of mineral soil and clay that dries into an opaque, matte crust with microscopic fissures, rough granular textures, and distinct perimeter borders.",
    "2. Chrominance Signatures in CIELAB: Even on grey cotton, used cooking oil exhibits a subtle amber-yellow chromatic shift (positive b* deflection), whereas dried mud exhibits an earthy brown hue with distinct a* and b* coordinate ratios.",
    "MobileNetV2's depthwise convolutional kernels capture these fine micro-textural gradients and chromatic transitions. In our 360-image validation evaluation, there were zero cross-misclassifications between mud and oil."
])

add_callout_box(doc, "Question 4: How did you prevent overfitting and data leakage during model training and evaluation?", [
    "Defense Answer: We enforced strict methodological controls throughout our machine learning lifecycle:",
    "1. Stratified Partitioning: We performed a stratified 80/20 train/validation split using scikit-learn with a fixed random seed (42). This guaranteed that each subset contained an identical ratio of classes and substrates (240 train / 60 val per class per substrate), eliminating distribution skew.",
    "2. Data Leakage Prevention: Data augmentations—such as random flips, brightness shifts (+/-25), contrast scaling (0.7-1.4), and HSV jitter—were executed on-the-fly strictly on the training pipeline via tf.data. The validation set was never augmented; it received only standard bilinear resizing, CLAHE, and MobileNetV2 normalization.",
    "3. Controlled Fine-Tuning: During Phase 1, the base model remained frozen. During Phase 2, only the top 20 layers were fine-tuned with a low learning rate (1e-5), while all BatchNormalization layers remained frozen to preserve global running statistics. A 40% Dropout rate on the classification head further prevented co-adaptation of weights."
])

add_callout_box(doc, "Question 5: How does the system handle edge cases in the mobile client, such as network failure or server cold starts on Render?", [
    "Defense Answer: Because our backend is deployed on Render's container tier, inactive instances spin down after inactivity, causing a cold-start delay of 30 to 45 seconds when a new request arrives. We resolved this through both server and client engineering:",
    "On the backend, Gunicorn runs with 4 asynchronous threads and pre-warms the model upon startup by passing a dummy tensor through the graph, pre-compiling the execution graph.",
    "On the mobile client (app.js), we built an asynchronous polling controller with a multi-step animated progress overlay in English and Hiligaynon. If a cold-start delay occurs, the client maintains the connection and presents informative status messages rather than timing out.",
    "If network connectivity fails completely, the client provides graceful degradation: it alerts the user, provides an offline simulation/testing mode, and retrieves cached knowledge base recipes stored in localStorage so the user is never left without cleaning assistance."
])

# =========================================================
# SECTION 5: QUICK DEFENSE CHECKLIST
# =========================================================
add_h1("5. Quick Defense Checklist & Summary")

checklist_items = [
    ("Model Backbone", "MobileNetV2 (154 layers, ImageNet weights, Top 20 layers fine-tuned)"),
    ("Classification Head", "GlobalAveragePooling2D -> BatchNormalization -> Dropout(0.40) -> Dense(3, softmax)"),
    ("Loss & Optimizer", "Categorical Crossentropy, Adam (Phase 1: 1e-3, Phase 2: 1e-5)"),
    ("Input Resolution", "224 x 224 x 3 RGB"),
    ("Key Preprocessing", "CIELAB CLAHE (L* channel, clipLimit=2.0, tileGrid=(8,8)) + preprocess_input ([-1.0, 1.0])"),
    ("Total Dataset", "1,800 physical photos (900 White Cotton, 900 Grey Cotton | 600 per class)"),
    ("Target Classes", "Used Cooking Oil, Black Ballpoint Pen Ink, Mud"),
    ("Validation Accuracy", "99.44% on 360 unseen test images (358 / 360 correct)"),
    ("Backend Architecture", "Dockerized Flask 3.0.3 + Gunicorn (1 worker, 4 threads, 120s timeout) on Render"),
    ("Database Configuration", "MongoDB Atlas (M0 Cluster) with PyMongo TLS + in-memory/JSON fallback"),
    ("Mobile Framework", "Hybrid web client (HTML5/CSS3/ES6 JS) packaged via Capacitor / WebIntoApp into Android APK"),
    ("Localization", "English and Hiligaynon (Ilonggo) bilingual support")
]

tbl_check = doc.add_table(rows=len(checklist_items)+1, cols=2)
tbl_check.alignment = WD_TABLE_ALIGNMENT.CENTER
widths_check = [2.2, 4.3]

tbl_check.cell(0, 0).paragraphs[0].text = "Technical Criterion"
tbl_check.cell(0, 1).paragraphs[0].text = "Verified Implementation Specification"
style_table_header(tbl_check.rows[0], widths_check)

for idx, (crit, spec) in enumerate(checklist_items, start=1):
    tbl_check.cell(idx, 0).paragraphs[0].text = f"[PASS] {crit}"
    tbl_check.cell(idx, 1).paragraphs[0].text = spec
    style_table_row(tbl_check.rows[idx], is_zebra=(idx % 2 == 0), col_widths=widths_check)

doc.save(DOCX_OUT)
print(f"Successfully generated Word document: {DOCX_OUT}")
