#!/usr/bin/env python
"""
Qwen2.5-VL Single Image Prediction Script
Predict if an image is an ECCO shoe or not.

Usage:
    python predict.py <image_path>
    python predict.py <image_path> --model_path <path_to_model>

Examples:
    python predict.py shoe.jpg
    python predict.py shoe.jpg --model_path qwen2.5-vl-lora-model/checkpoint-50
"""

import os
import sys
import argparse
import torch
from PIL import Image
from unsloth import FastVisionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Predict if an image is an ECCO shoe")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to analyze"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen2.5-vl-lora-model",
        help="Path to the trained model checkpoint (default: qwen2.5-vl-lora-model)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Is this an ECCO shoe?",
        help="Prompt to use for prediction"
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1024,
        help="Maximum image size (larger images will be resized)"
    )
    return parser.parse_args()


def process_image(image_path, max_size=1024):
    """Opens image and resizes it if needed."""
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


def predict(image_path, model, tokenizer, prompt, max_image_size):
    """Run prediction on a single image."""
    # Load and process image
    pil_image = process_image(image_path, max_image_size)
    
    # Prepare prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
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

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0,
            do_sample=False,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    return response


def main():
    args = parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("ECCO Shoe Detection")
    print("=" * 50)
    print(f"Image: {args.image_path}")
    print(f"Model: {args.model_path}")
    print()
    
    # Load model
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded!")
    print()
    
    # Run prediction
    print("Analyzing image...")
    response = predict(args.image_path, model, tokenizer, args.prompt, args.max_image_size)
    
    # Display result
    print("=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"Response: {response}")
    print()
    
    # Simple classification
    response_lower = response.lower()
    if "yes" in response_lower:
        print("✅ This IS an ECCO shoe")
    else:
        print("❌ This is NOT an ECCO shoe")


if __name__ == "__main__":
    main()
