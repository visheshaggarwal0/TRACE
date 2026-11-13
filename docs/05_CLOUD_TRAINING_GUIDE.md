# TRACE: Cloud Training Guide (Google Colab & Kaggle GPU)

**Document ID:** `TRACE-DOC-005`  
**Version:** `1.0.0`  
**System Name:** **TRACE** (*Topological Road Augmentation and Completion Engine*)

---

## 1. Overview

Training deep convolutional models (like D-LinkNet34) on high-resolution satellite imagery tiles ($512 \times 512$ or $1024 \times 1024$) is compute-heavy. While local laptops with CPU or low-VRAM GPUs are great for code debugging and unit tests, full model training should be run on **free cloud GPUs** via **Google Colab** or **Kaggle**.

| Cloud Platform | Free GPU Available | Max Session Time | Best Used For |
|---|---|---|---|
| **Google Colab** | 1x NVIDIA T4 (16 GB VRAM) | 12 Hours | Fast prototyping, Google Drive checkpoint syncing |
| **Kaggle Notebooks** | 2x NVIDIA T4 (30 hours/week) | 12 Hours | Large-scale multi-GPU training on SpaceNet datasets |

---

## 2. Option A: Training on Google Colab (Recommended)

### Step 1: Open a New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **New Notebook**.
3. In the menu, go to **Runtime** $\to$ **Change runtime type** $\to$ Select **T4 GPU** $\to$ Click **Save**.

### Step 2: Mount Google Drive (To Save Checkpoints Permanently)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Clone Your TRACE Repository
```bash
# Clone repository from your GitHub or copy from Drive
!git clone https://github.com/<your-username>/TRACE.git /content/TRACE
%cd /content/TRACE
```

### Step 4: Install Dependencies
```bash
!pip install -q -r requirements.txt
!pip install -q pytest
```

### Step 5: Verify GPU & Run Unit Tests
```bash
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Run unit tests
!pytest tests/ -v
```

### Step 6: Generate Sample Data or Download SpaceNet 3
```bash
# Option 1: Fast synthetic tiles for pipeline verification
!python scripts/generate_synthetic_sample.py

# Option 2: If using SpaceNet 3, symlink or copy from your Google Drive
# !cp -r /content/drive/MyDrive/SpaceNet3_Data/ /content/TRACE/data/raw/
```

### Step 7: Launch Training with Automatic Checkpoint Sync to Drive
```bash
# Create checkpoints directory linked to Google Drive
!mkdir -p /content/drive/MyDrive/TRACE_Checkpoints/

# Run training
!python train.py --epochs 60 --batch_size 16 --lr 0.0002
```

### Step 8: Run Evaluation & Download Final Model
```bash
# Run APLS evaluation
!python evaluate.py --checkpoint checkpoints/best_model.pth

# Copy best weights to your permanent Google Drive storage
!cp checkpoints/best_model.pth /content/drive/MyDrive/TRACE_Checkpoints/best_model_spacenet3.pth
```

---

## 3. Option B: Training on Kaggle Notebooks

Kaggle provides **30 hours per week of free dual-T4 GPUs** and has SpaceNet datasets hosted publicly:

### Step 1: Create a Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com) $\to$ **Code** $\to$ **New Notebook**.
2. In the right-hand panel:
   * **Accelerator:** Select **GPU T4 x2**.
   * **Internet:** Toggle to **On**.

### Step 2: Add SpaceNet Dataset Directly (Zero Download Time!)
1. In the right panel, click **+ Add Input**.
2. Search for `SpaceNet 3 Road Network` or `DeepGlobe Road Extraction`.
3. Click **Add** to attach the dataset instantly to `/kaggle/input/`.

### Step 3: Clone TRACE and Train
```bash
!git clone https://github.com/<your-username>/TRACE.git /kaggle/working/TRACE
%cd /kaggle/working/TRACE

!pip install -q -r requirements.txt

# Run training pointing raw_dir to the Kaggle dataset
!python train.py --config config/default_config.yaml --epochs 50 --batch_size 16
```

---

## 4. Best Practices for Cloud Training

1. **Avoid Session Disconnects:** Keep the Colab tab active.
2. **Always Save Checkpoints to Drive:** Never save `.pth` files only to `/content/`—if Colab disconnects, local files are deleted. Always save or copy to `/content/drive/MyDrive/`.
3. **Use FP16 Mixed Precision:** In `train.py`, if training on a GPU, use `torch.cuda.amp.autocast()` to double your training speed and halve VRAM usage:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       logits = model(inputs)
       loss = criterion(logits, targets)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```
