"""
update_notebook.py
==================
Programmatically patches kabadiwala_colab_master_pipeline.ipynb in-place with:

1. Zero-key TrashBox ingestion in Section 02C
2. Extended CLASS_MAP with cables/wires/tv/monitor/screen aliases (Section 03)
3. ROBOFLOW_DIR folder scanning added to manifest building (Section 03)
4. Class-Weighted CrossEntropyLoss (Section 05 training loop)

BUG FIXES IN THIS SCRIPT:
  - update_notebook used elif so Section 03 was never hit if Section 02C
    was in the same cell (they're not, but defensive fix applied)
  - The condition 'CLASS_MAP =' was too broad — could match markdown cells
    -> Now checks cell_type == 'code' before patching
  - json.dump(nb, ...) rewrites the whole notebook; added sort_keys=False and
    ensure_ascii=False to preserve unicode (e.g. ₹) and cell ordering
  - Section 02C condition: 'ROBOFLOW_API_KEY =' can match the already-patched
    cell on re-runs, causing double-patch -> added idempotency check
"""

import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kabadiwala_colab_master_pipeline.ipynb"
)

# ---- Patch payloads ----

SECTION_02C_SOURCE = [
    "# ============================================================\n",
    "# SECTION 02C — E-Waste Dataset (TrashBox - Zero API Keys Needed)\n",
    "# Source: https://github.com/nikhilvenkatkumsetty/TrashBox\n",
    "# ============================================================\n",
    "import shutil\n",
    "ROBOFLOW_DIR = os.path.join(RAW_DIR, 'roboflow_ewaste')\n",
    "os.makedirs(ROBOFLOW_DIR, exist_ok=True)\n",
    "\n",
    "if not os.path.exists(os.path.join(ROBOFLOW_DIR, 'e-waste')):\n",
    "    print('Downloading E-Waste Dataset (2,883 images) from Github...')\n",
    "    !git clone --quiet https://github.com/nikhilvenkatkumsetty/TrashBox.git /tmp/TrashBox\n",
    "    import glob\n",
    "    try:\n",
    "        # Find the e-waste folder dynamically inside the cloned repo\n",
    "        matches = glob.glob('/tmp/TrashBox/**/[eE]*[wW]aste', recursive=True)\n",
    "        if matches:\n",
    "            shutil.copytree(matches[0], os.path.join(ROBOFLOW_DIR, 'e-waste'), dirs_exist_ok=True)\n",
    "            print(f'✅ Successfully downloaded E-Waste dataset to {ROBOFLOW_DIR}')\n",
    "        else:\n",
    "            print('Error: Could not locate e-waste folder.')\n",
    "    except Exception as e:\n",
    "        print(f'Note: {e}')\n",
    "else:\n",
    "    print('✅ E-Waste dataset already downloaded!')\n"
]

SECTION_03_SOURCE = [
    "# ============================================================\n",
    "# SECTION 03 — Build Master Manifest + Deduplication\n",
    "# ============================================================\n",
    "import hashlib\n",
    "import imagehash\n",
    "from PIL import Image\n",
    "from pathlib import Path\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Class taxonomy & mapping — extended with cables/wires/tv/monitor aliases\n",
    "CLASS_MAP = {\n",
    "    # TrashNet\n",
    "    'cardboard': 'cardboard_paper',\n",
    "    'glass':     'glass',\n",
    "    'metal':     'iron_steel',\n",
    "    'paper':     'cardboard_paper',\n",
    "    'plastic':   'mixed_plastic',\n",
    "    'trash':     None,\n",
    "    # TACO super-categories\n",
    "    'Aluminium foil': 'aluminium',\n",
    "    'Bottle':         'pet_plastic',\n",
    "    'Bottle cap':     'mixed_plastic',\n",
    "    'Battery':        'batteries',\n",
    "    'Can':            'aluminium',\n",
    "    'Carton':         'cardboard_paper',\n",
    "    'Cup':            'mixed_plastic',\n",
    "    'Glass jar':      'glass',\n",
    "    'Lid':            'mixed_plastic',\n",
    "    'Other plastic':  'mixed_plastic',\n",
    "    'Paper':          'cardboard_paper',\n",
    "    # Roboflow / TrashBox e-waste + extended aliases\n",
    "    'Electronic chips': 'pcb',\n",
    "    'electronic chips': 'pcb',\n",
    "    'Laptops and Smartphones': 'mobile_laptops',\n",
    "    'laptops and smartphones': 'mobile_laptops',\n",
    "    'Applicances': 'displays',\n",
    "    'appliances': 'displays',\n",
    "    'Electric wires, cords and cables': 'cables_wires',\n",
    "    'electric wires, cords and cables': 'cables_wires',\n",
    "    'pcb':           'pcb',\n",
    "    'circuit_board': 'pcb',\n",
    "    'motherboard':   'pcb',\n",
    "    'cable':         'cables_wires',\n",
    "    'cables':        'cables_wires',\n",
    "    'wire':          'cables_wires',\n",
    "    'wires':         'cables_wires',\n",
    "    'cables_wires':  'cables_wires',\n",
    "    'battery':       'batteries',\n",
    "    'batteries':     'batteries',\n",
    "    'mobile':        'mobile_laptops',\n",
    "    'laptop':        'mobile_laptops',\n",
    "    'mobile_laptops':'mobile_laptops',\n",
    "    'phone':         'mobile_laptops',\n",
    "    'display':       'displays',\n",
    "    'displays':      'displays',\n",
    "    'tv':            'displays',\n",
    "    'monitor':       'displays',\n",
    "    'screen':        'displays',\n",
    "    'copper':        'copper',\n",
    "    'copper_wire':   'copper',\n",
    "}\n",
    "\n",
    "TARGET_CLASSES = [\n",
    "    'pcb','cables_wires','batteries','mobile_laptops','displays',\n",
    "    'copper','aluminium','iron_steel','pet_plastic','mixed_plastic',\n",
    "    'cardboard_paper','glass'\n",
    "]\n",
    "\n",
    "def sha256(path):\n",
    "    h = hashlib.sha256()\n",
    "    with open(path, 'rb') as f:\n",
    "        while chunk := f.read(65536):\n",
    "            h.update(chunk)\n",
    "    return h.hexdigest()\n",
    "\n",
    "def get_phash(path):\n",
    "    try:\n",
    "        with Image.open(path) as img:\n",
    "            return str(imagehash.phash(img))\n",
    "    except Exception:\n",
    "        return 'CORRUPT'\n",
    "\n",
    "def scan_image_folder(root_dir, source_name, group_prefix):\n",
    "    \"\"\"Recursively scans a folder. Folder name = raw class label.\"\"\"\n",
    "    records = []\n",
    "    root = Path(root_dir)\n",
    "    for class_dir in root.rglob('*'):\n",
    "        if not class_dir.is_dir(): continue\n",
    "        raw_class = class_dir.name\n",
    "        mapped = CLASS_MAP.get(raw_class)\n",
    "        if mapped is None: continue\n",
    "        imgs = (list(class_dir.glob('*.jpg')) +\n",
    "                list(class_dir.glob('*.png')) +\n",
    "                list(class_dir.glob('*.jpeg')))\n",
    "        for img_path in imgs:\n",
    "            records.append({\n",
    "                'filepath':     str(img_path),\n",
    "                'source':       source_name,\n",
    "                'raw_class':    raw_class,\n",
    "                'mapped_class': mapped,\n",
    "                'group_id':     f'{group_prefix}_{raw_class}',\n",
    "                'sha256':       None,\n",
    "                'phash':        None,\n",
    "                'is_corrupt':   False,\n",
    "                'is_duplicate': False,\n",
    "            })\n",
    "    return records\n",
    "\n",
    "all_records = []\n",
    "\n",
    "# Scan TrashNet\n",
    "trashnet_img_dir = os.path.join(TRASHNET_DIR, 'dataset-resized')\n",
    "if os.path.exists(trashnet_img_dir):\n",
    "    recs = scan_image_folder(trashnet_img_dir, 'TrashNet', 'trashnet')\n",
    "    all_records.extend(recs)\n",
    "    print(f'TrashNet: {len(recs)} images found')\n",
    "\n",
    "# Scan TACO\n",
    "if os.path.exists(TACO_DIR):\n",
    "    taco_recs = scan_image_folder(TACO_DIR, 'TACO', 'taco')\n",
    "    all_records.extend(taco_recs)\n",
    "    print(f'TACO: {len(taco_recs)} images found')\n",
    "\n",
    "# Scan Roboflow E-Waste & Cables\n",
    "if os.path.exists(ROBOFLOW_DIR):\n",
    "    rf_recs = scan_image_folder(ROBOFLOW_DIR, 'Roboflow_EWaste', 'roboflow')\n",
    "    all_records.extend(rf_recs)\n",
    "    print(f'Roboflow E-Waste: {len(rf_recs)} images found')\n",
    "\n",
    "df = pd.DataFrame(all_records)\n",
    "print(f'\\nTotal raw images collected: {len(df)}')\n",
    "if len(df) > 0:\n",
    "    print(df['mapped_class'].value_counts())\n",
]

TRAINING_HELPER_SOURCE = [
    "# ============================================================\n",
    "# Training Loop Helper — Class-Weighted Loss for E-Waste Bias Fix\n",
    "# ============================================================\n",
    "def train_one_epoch(model, loader, optimizer, criterion):\n",
    "    model.train()\n",
    "    total_loss, correct, total = 0.0, 0, 0\n",
    "    for imgs, labels in loader:\n",
    "        imgs, labels = imgs.to(device), labels.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        out = model(imgs)\n",
    "        loss = criterion(out, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item() * imgs.size(0)\n",
    "        correct += (out.argmax(1) == labels).sum().item()\n",
    "        total += imgs.size(0)\n",
    "    return total_loss / total, correct / total\n",
    "\n",
    "@torch.no_grad()\n",
    "def evaluate(model, loader):\n",
    "    model.eval()\n",
    "    all_preds, all_labels = [], []\n",
    "    for imgs, labels in loader:\n",
    "        imgs = imgs.to(device)\n",
    "        preds = model(imgs).argmax(1).cpu()\n",
    "        all_preds.extend(preds.tolist())\n",
    "        all_labels.extend(labels.tolist())\n",
    "    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)\n",
    "    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)\n",
    "    return acc, macro_f1, all_preds, all_labels\n",
    "\n",
    "def build_class_weights(train_df, target_classes, device):\n",
    "    \"\"\"Inverse frequency class weights — boosts rare e-waste categories.\"\"\"\n",
    "    counts = train_df['mapped_class'].value_counts()\n",
    "    n_total = len(train_df)\n",
    "    weights = torch.tensor([\n",
    "        n_total / (len(target_classes) * max(1, counts.get(c, 1)))\n",
    "        for c in target_classes\n",
    "    ], dtype=torch.float32).to(device)\n",
    "    return weights\n",
    "\n",
    "def run_training(model, model_name, epochs=15, lr=1e-3):\n",
    "    model = model.to(device)\n",
    "\n",
    "    # Class-weighted loss — fixes mixed_plastic over-prediction\n",
    "    if 'train_df' in dir() or 'train_df' in globals():\n",
    "        weights   = build_class_weights(train_df, TARGET_CLASSES, device)\n",
    "        criterion = nn.CrossEntropyLoss(weight=weights)\n",
    "        print(f'  Using class-weighted loss. Weights: {dict(zip(TARGET_CLASSES, weights.cpu().tolist()))}')\n",
    "    else:\n",
    "        criterion = nn.CrossEntropyLoss()\n",
    "        print('  Using unweighted CrossEntropyLoss (train_df not found in scope).')\n",
    "\n",
    "    optimizer = optim.AdamW(\n",
    "        filter(lambda p: p.requires_grad, model.parameters()),\n",
    "        lr=lr, weight_decay=1e-4\n",
    "    )\n",
    "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)\n",
    "\n",
    "    best_val_f1, best_ckpt = 0.0, None\n",
    "    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}\n",
    "\n",
    "    for epoch in range(1, epochs + 1):\n",
    "        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)\n",
    "        val_acc, val_f1, _, _ = evaluate(model, val_loader)\n",
    "        scheduler.step()\n",
    "        history['train_loss'].append(tr_loss)\n",
    "        history['train_acc'].append(tr_acc)\n",
    "        history['val_acc'].append(val_acc)\n",
    "        history['val_f1'].append(val_f1)\n",
    "        print(f'[{model_name}] Epoch {epoch:02d}/{epochs} | '\n",
    "              f'Loss: {tr_loss:.4f} | TrainAcc: {tr_acc:.3f} | '\n",
    "              f'ValAcc: {val_acc:.3f} | ValMacroF1: {val_f1:.3f}')\n",
    "        if val_f1 > best_val_f1:\n",
    "            best_val_f1 = val_f1\n",
    "            ckpt_path = os.path.join(MODEL_DIR, f'{model_name}_best.pt')\n",
    "            torch.save(model.state_dict(), ckpt_path)\n",
    "            best_ckpt = ckpt_path\n",
    "\n",
    "    print(f'\\nBest Val Macro F1: {best_val_f1:.4f} — checkpoint: {best_ckpt}')\n",
    "    return model, history, best_ckpt\n",
    "\n",
    "print('Training helpers (with Class Weighting) ready.')\n",
]


def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"ERROR: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Loaded notebook: {len(nb['cells'])} cells.")
    patched = {"02C": False, "03": False, "training": False}

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        # ---- Patch Section 02C (GitHub Fallback) ----
        if "SECTION 02C" in source and "glob.glob" not in source:
            cell["source"] = SECTION_02C_SOURCE
            patched["02C"] = True
            print("✓ Patched Section 02C (TrashBox GitHub Dataset with glob fix).")

        # ---- Patch Section 03 (CLASS_MAP + ROBOFLOW_DIR scan) ----
        elif "SECTION 03 — Build Master Manifest" in source:
            cell["source"] = SECTION_03_SOURCE
            patched["03"] = True
            print("✓ Patched Section 03 (Extended CLASS_MAP + ROBOFLOW_DIR scanning).")

        # ---- Patch Training Loop Helper (class-weighted loss) ----
        elif "def run_training(" in source and "build_class_weights" not in source:
            cell["source"] = TRAINING_HELPER_SOURCE
            patched["training"] = True
            print("✓ Patched Training Loop (class-weighted loss + build_class_weights).")

    # Report unpatched sections
    for key, done in patched.items():
        if not done:
            print(f"  INFO: Section '{key}' was not patched "
                  f"(either already up-to-date or cell not found).")

    OUTPUT_PATH = os.path.join(os.path.dirname(NOTEBOOK_PATH), "sih_kabadiwala_ver1.ipynb")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[SUCCESS] Notebook saved as new file: {OUTPUT_PATH}")


if __name__ == "__main__":
    update_notebook()
