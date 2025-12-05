# Qwen2.5-VL Vision Language Model Training

This folder contains everything needed to fine-tune the Qwen2.5-VL model for ECCO shoe detection.

## Files

- `train_qwen25_vl.py` - Main training script
- `evaluate.py` - Evaluation script for batch testing on test dataset
- `predict.py` - Single image prediction script
- `requirements.txt` - Python dependencies
- `dataset/` - Training dataset with images and train.json
- `test_dataset/` - Test dataset for evaluation (ecco/ and not_ecco/ subfolders)


## Dataset Setup

## Dataset Setup

## Dataset Setup

The repository includes two RAR archives containing the dataset:
1. `images.rar` (Training images)
2. `test_dataset.rar` (Evaluation images)

### Setup Steps:

1. **Extract Training Data**:
   - Extract `images.rar` into the `dataset/` folder.
   - You should end up with `dataset/images/`.

2. **Extract Test Data**:
   - Extract `test_dataset.rar` into the root folder.
   - You should end up with `test_dataset/ecco/` and `test_dataset/not_ecco/`.

After extraction, your folder structure should look like this:
```
qwen25_vl_training/
├── dataset/
│   ├── images/         <-- Contains training images
│   └── train.json      <-- Dataset JSON
├── test_dataset/
│   ├── ecco/           <-- Contains ECCO test images
│   └── not_ecco/       <-- Contains Non-ECCO test images
```

## Quick Start

### 1. Install PyTorch with CUDA Support

First, check your CUDA version:
```bash
nvidia-smi
```

Then install PyTorch with your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/):
```bash
# Example for CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
# Default training (70 steps as in notebook)
python train_qwen25_vl.py

# Custom max steps
python train_qwen25_vl.py --max_steps 100

# Custom output directory
python train_qwen25_vl.py --output_dir my_model
```

### 4. Evaluate the Model

```bash
# Evaluate final saved model (default)
python evaluate.py

# Evaluate a specific checkpoint (e.g., checkpoint 50)
python evaluate.py --model_path qwen2.5-vl-lora-model/checkpoint-50

# Evaluate with custom test directory
python evaluate.py --model_path qwen2.5-vl-lora-model/checkpoint-50 --test_dir ./test_data
```

### 5. Single Image Prediction

```bash
# Predict on a single image
python predict.py path/to/shoe.jpg

# Use a specific checkpoint
python predict.py path/to/shoe.jpg --model_path qwen2.5-vl-lora-model/checkpoint-50
```

## Checkpoint Locations

After training, checkpoints are saved in the output directory:

| Checkpoint | Path |
|------------|------|
| Final model | `qwen2.5-vl-lora-model/` |
| Step 10 | `qwen2.5-vl-lora-model/checkpoint-10/` |
| Step 20 | `qwen2.5-vl-lora-model/checkpoint-20/` |
| Step 30 | `qwen2.5-vl-lora-model/checkpoint-30/` |
| Step 40 | `qwen2.5-vl-lora-model/checkpoint-40/` |
| Step 50 | `qwen2.5-vl-lora-model/checkpoint-50/` |
| Step 60 | `qwen2.5-vl-lora-model/checkpoint-60/` |
| Step 70 | `qwen2.5-vl-lora-model/checkpoint-70/` |

**Note:** Use `--output_dir` during training to change the checkpoint location.

## Training Parameters (from notebook)

| Parameter | Default Value |
|-----------|---------------|
| max_steps | 70 |
| learning_rate | 5e-5 |
| batch_size | 2 |
| gradient_accumulation | 4 |
| save_steps | 10 |
| lora_r | 16 |
| lora_alpha | 16 |

## Dataset Format

The dataset (`dataset/train.json`) should follow this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "dataset/images/image.jpg"},
        {"type": "text", "text": "Is this an ECCO shoe?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Yes, this is an ECCO shoe."}
      ]
    }
  ]
}
```

## Test Dataset Structure (for evaluation)

```
test_dataset/
├── ecco/           # ECCO shoe images
│   ├── img1.jpg
│   └── img2.jpg
└── not_ecco/       # Non-ECCO shoe images
    ├── img3.jpg
    └── img4.jpg
```
