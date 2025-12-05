#!/usr/bin/env python
"""
Qwen2.5-VL Vision Language Model Training Script
Based on the Qwen2_5_VL_(7B)_Vision.ipynb notebook

This script fine-tunes the Qwen2.5-VL model using LoRA adapters for vision-language tasks.
It uses the unsloth library for efficient training and the TRL library for supervised fine-tuning.

Usage:
    python train_qwen25_vl.py [options]

Examples:
    # Default training (70 steps as in notebook)
    python train_qwen25_vl.py

    # Custom max steps
    python train_qwen25_vl.py --max_steps 100

    # Custom output directory
    python train_qwen25_vl.py --output_dir my_model
"""

import os
import json
import argparse
from PIL import Image

# Fix for Windows Triton compilation errors
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# Disable the fast (but sometimes buggy) Rust downloader
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL model with LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default training (70 steps as in notebook)
    python train_qwen25_vl.py
    
    # Custom max steps
    python train_qwen25_vl.py --max_steps 100
    
    # Custom output directory
    python train_qwen25_vl.py --output_dir my_model
        """
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/train.json",
        help="Path to the training dataset JSON file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        help="Model name or path"
    )
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true", 
        default=True,
        help="Load model in 4-bit quantization"
    )
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=70,
        help="Maximum training steps (default: 70 as in notebook)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Learning rate (default: 5e-5 as in notebook)"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=10,
        help="Warmup steps"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048,
        help="Maximum sequence length"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="qwen2.5-vl-lora-model",
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=10,
        help="Save checkpoint every N steps (default: 10 as in notebook)"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=1,
        help="Log every N steps (default: 1 as in notebook)"
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=10,
        help="Maximum number of checkpoints to keep (default: 10 as in notebook)"
    )
    
    return parser.parse_args()


def load_dataset(dataset_path: str):
    """Load dataset from JSON file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def load_images_for_training(data: list):
    """Load actual PIL images for each data entry."""
    print("Loading images from disk...")
    processed_data = []
    
    for item in data:
        processed_item = {"messages": []}
        
        for message in item["messages"]:
            processed_message = {"role": message["role"], "content": []}
            
            for content in message["content"]:
                if content["type"] == "image":
                    img_path = content["image"]
                    try:
                        img = Image.open(img_path)
                        # Convert RGBA to RGB if necessary
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        processed_message["content"].append({"type": "image", "image": img})
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}: {e}")
                        continue
                else:
                    processed_message["content"].append(content)
            
            processed_item["messages"].append(processed_message)
        
        processed_data.append(processed_item)
    
    print(f"Processed {len(processed_data)} examples")
    return processed_data


def load_model(model_name: str, load_in_4bit: bool = True):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"4-bit quantization: {load_in_4bit}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    
    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 16, dropout: float = 0):
    """Apply LoRA adapters to the model."""
    print(f"Applying LoRA adapters (r={r}, alpha={alpha}, dropout={dropout})")
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model


def create_trainer(
    model,
    tokenizer,
    train_dataset: list,
    output_dir: str = "outputs",
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 10,
    max_steps: int = 70,
    save_steps: int = 10,
    logging_steps: int = 1,
    max_length: int = 2048,
    save_total_limit: int = 10,
):
    """Create the SFTTrainer."""
    print("\nCreating trainer...")
    
    # Enable training mode
    FastVisionModel.for_training(model)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_length,
        ),
    )
    
    return trainer


def print_training_config(
    train_dataset: list,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 70,
    learning_rate: float = 5e-5,
):
    """Print training configuration."""
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // effective_batch_size
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60 + "\n")


def print_memory_stats():
    """Print GPU memory statistics."""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU: {gpu_stats.name}")
        print(f"Max memory: {max_memory} GB")
        print(f"Peak reserved memory: {used_memory} GB")
        print(f"Memory usage: {round(used_memory / max_memory * 100, 2)}%")


def save_model(model, tokenizer, output_dir: str):
    """Save the trained model and tokenizer."""
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("Qwen2.5-VL Vision Language Model Training")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Training will be slow.")
    
    # Load dataset
    train_data = load_dataset(args.dataset_path)
    
    # Convert images to PIL format
    train_dataset = load_images_for_training(train_data)
    
    # Load model
    model, tokenizer = load_model(args.model_name, args.load_in_4bit)
    
    # Apply LoRA
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Print configuration
    print_training_config(
        train_dataset,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_steps,
        args.learning_rate,
    )
    
    # Create trainer
    trainer = create_trainer(
        model,
        tokenizer,
        train_dataset,
        args.output_dir,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.warmup_steps,
        args.max_steps,
        args.save_steps,
        args.logging_steps,
        args.max_length,
        args.save_total_limit,
    )
    
    # Print initial memory stats
    print_memory_stats()
    
    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()
    
    # Print training stats
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print("=" * 60)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes")
    
    # Print memory stats
    print_memory_stats()
    
    # Save model
    save_model(model, tokenizer, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training script completed successfully!")
    print("=" * 60)
    print(f"\nTo load the trained model for inference:")
    print(f"  from unsloth import FastVisionModel")
    print(f"  model, tokenizer = FastVisionModel.from_pretrained('{args.output_dir}')")
    print(f"  FastVisionModel.for_inference(model)")


if __name__ == "__main__":
    main()
