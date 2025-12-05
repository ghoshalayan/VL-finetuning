#!/usr/bin/env python
"""
Qwen2.5-VL Model Evaluation Script
Evaluates the fine-tuned model on a test dataset.

Usage:
    python evaluate.py [--model_path PATH] [--test_dir DIR]

Examples:
    # Default evaluation
    python evaluate.py
    
    # Custom model path
    python evaluate.py --model_path qwen2.5-vl-lora-model
    
    # Custom test directory
    python evaluate.py --test_dir ./my_test_data
"""

import os
import argparse
import torch
from unsloth import FastVisionModel
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Qwen2.5-VL model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen2.5-vl-lora-model",
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_dataset",
        help="Directory containing test images (with 'ecco' and 'not_ecco' subfolders)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Is this an ECCO shoe?",
        help="Prompt to use for evaluation"
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1024,
        help="Maximum image size (larger images will be resized)"
    )
    return parser.parse_args()


def process_image(image_path, max_size=1024):
    """Opens image and resizes it to max size to match production"""
    img = Image.open(image_path).convert("RGB")
    
    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Qwen2.5-VL Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded successfully!")
    
    # Initialize lists for predictions
    y_true = []
    y_pred = []
    
    def predict(image_path):
        try:
            # Resize Image
            pil_image = process_image(image_path, args.max_image_size)
            
            # Prepare Prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": args.prompt}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Tokenize
            inputs = tokenizer(
                pil_image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            # Generate (Deterministic)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10, 
                    temperature=0,     
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            
            # Parse Answer
            if "yes" in response:
                return 1
            return 0
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0  # Default to 'No' on error

    # Process ECCO Folder (Ground Truth = 1)
    ecco_path = os.path.join(args.test_dir, "ecco")
    print("\nTesting ECCO shoes...")
    if os.path.exists(ecco_path):
        for img in tqdm(os.listdir(ecco_path)):
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                y_true.append(1)
                pred = predict(os.path.join(ecco_path, img))
                y_pred.append(pred)
    else:
        print(f"Warning: ECCO folder not found at {ecco_path}")

    # Process NOT_ECCO Folder (Ground Truth = 0)
    not_ecco_path = os.path.join(args.test_dir, "not_ecco")
    print("\nTesting OTHER shoes...")
    if os.path.exists(not_ecco_path):
        for img in tqdm(os.listdir(not_ecco_path)):
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                y_true.append(0)
                pred = predict(os.path.join(not_ecco_path, img))
                y_pred.append(pred)
    else:
        print(f"Warning: Not-ECCO folder not found at {not_ecco_path}")

    # Report
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if len(y_true) > 0:
        correct = sum([1 for t, p in zip(y_true, y_pred) if t == p])
        total = len(y_true)
        print(f"\nTotal Accuracy: {correct/total*100:.2f}% ({correct}/{total})")

        print("\nDetailed Metrics:")
        target_names = ['Not Ecco', 'Ecco']
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        try:
            print(f"  True Negatives (Correctly said No):  {cm[0][0]}")
            print(f"  False Positives (Wrongly said Yes):  {cm[0][1]}") 
            print(f"  False Negatives (Wrongly said No):   {cm[1][0]}")
            print(f"  True Positives (Correctly said Yes): {cm[1][1]}")
        except IndexError:
            print("  Not enough classes to display full confusion matrix.")
    else:
        print("\nNo images found in test folders.")
        print(f"Expected structure:")
        print(f"  {args.test_dir}/")
        print(f"    ecco/       (ECCO shoe images)")
        print(f"    not_ecco/   (Non-ECCO shoe images)")


if __name__ == "__main__":
    main()