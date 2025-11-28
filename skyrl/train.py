#!/usr/bin/env python3
"""
SkyRL training script with PPO agent and LoRA fine-tuning for Qwen models.
Uses GSM8K dataset for initial testing.
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

# Try importing SkyRL components
try:
    import skyrl
    from skyrl import SkyAgent, SkyTrainer
except ImportError:
    print("Warning: SkyRL not installed. Some features may not work.")
    skyrl = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_with_lora(
    model_name: str,
    lora_config: Dict[str, Any],
    device_map: str = "auto"
) -> tuple:
    """
    Load model and apply LoRA fine-tuning configuration.
    
    Args:
        model_name: HuggingFace model name
        lora_config: LoRA configuration dictionary
        device_map: Device mapping for model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Apply LoRA if enabled
    if lora_config.get("enabled", True):
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias=lora_config.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Warning: LoRA disabled. This will update all model weights!")
    
    return model, tokenizer


def load_gsm8k_dataset(data_dir: str) -> tuple:
    """
    Load prepared GSM8K dataset.
    
    Args:
        data_dir: Directory containing prepared dataset
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    import json
    
    data_path = Path(data_dir)
    train_file = data_path / "train.jsonl"
    eval_file = data_path / "eval.jsonl"
    
    def load_jsonl(file_path):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(eval_file)
    
    print(f"Loaded {len(train_data)} train examples")
    print(f"Loaded {len(eval_data)} eval examples")
    
    return train_data, eval_data


def setup_wandb(wandb_config: Dict[str, Any], run_name: Optional[str] = None):
    """Initialize WandB logging."""
    if wandb_config.get("enabled", False):
        wandb.init(
            project=wandb_config.get("project", "rft-llm-skyrl"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []),
            name=run_name,
        )


def main():
    parser = argparse.ArgumentParser(description="Train Qwen model with SkyRL PPO + LoRA")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--qwen-config",
        type=str,
        default="configs/qwen_config.yaml",
        help="Path to Qwen model configuration file"
    )
    parser.add_argument(
        "--gsm8k-config",
        type=str,
        default="configs/gsm8k_config.yaml",
        help="Path to GSM8K dataset configuration file"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Qwen model size to use (qwen2_0_5b, qwen2_1_5b, qwen2_7b, qwen2_72b)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB run name"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    print("Loading configurations...")
    base_config = load_config(args.base_config)
    qwen_config = load_config(args.qwen_config)
    gsm8k_config = load_config(args.gsm8k_config)
    
    # Determine model to use
    if args.model_size:
        model_key = args.model_size
    else:
        model_key = qwen_config.get("default_model", "qwen2_1_5b")
    
    if model_key not in qwen_config["models"]:
        raise ValueError(f"Unknown model size: {model_key}")
    
    model_info = qwen_config["models"][model_key]
    model_name = model_info["model_name"]
    
    print(f"Using model: {model_name}")
    
    # Initialize WandB
    setup_wandb(base_config.get("wandb", {}), args.run_name)
    
    # Load model with LoRA
    model, tokenizer = load_model_with_lora(
        model_name=model_name,
        lora_config=base_config.get("lora", {}),
    )
    
    # Load dataset
    data_dir = gsm8k_config["dataset"]["preparation"]["output_dir"]
    train_data, eval_data = load_gsm8k_dataset(data_dir)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=base_config["output"]["checkpoint_dir"],
        num_train_epochs=base_config["training"]["num_epochs"],
        per_device_train_batch_size=base_config["training"]["batch_size"],
        gradient_accumulation_steps=base_config["training"]["gradient_accumulation_steps"],
        learning_rate=base_config["training"]["learning_rate"],
        warmup_steps=base_config["training"]["warmup_steps"],
        max_grad_norm=base_config["training"]["max_grad_norm"],
        save_steps=base_config["training"]["save_steps"],
        eval_steps=base_config["training"]["eval_steps"],
        logging_steps=base_config["training"]["logging_steps"],
        logging_dir=base_config["output"]["log_dir"],
        report_to="wandb" if base_config.get("wandb", {}).get("enabled", False) else None,
        bf16=True,  # Use bfloat16 for H100 GPUs
        gradient_checkpointing=True,
        save_total_limit=3,
        load_best_model_at_end=True,
    )
    
    # Setup DeepSpeed if enabled
    if base_config.get("deepspeed", {}).get("enabled", False):
        deepspeed_config = {
            "zero_optimization": {
                "stage": base_config["deepspeed"]["zero_stage"],
            }
        }
        if base_config["deepspeed"].get("offload_optimizer", False):
            deepspeed_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu"
            }
        if base_config["deepspeed"].get("offload_param", False):
            deepspeed_config["zero_optimization"]["offload_param"] = {
                "device": "cpu"
            }
        training_args.deepspeed = deepspeed_config
    
    # TODO: Integrate SkyRL PPO trainer here
    # This is a placeholder structure - actual SkyRL integration will depend on
    # SkyRL's API. The structure should follow:
    # 1. Create SkyRL environment from GSM8K dataset
    # 2. Initialize PPO agent with model
    # 3. Run training loop with PPO updates
    
    print("\nTraining setup complete!")
    print(f"Model: {model_name}")
    print(f"LoRA enabled: {base_config.get('lora', {}).get('enabled', True)}")
    print(f"Train examples: {len(train_data)}")
    print(f"Eval examples: {len(eval_data)}")
    print("\nNote: SkyRL PPO trainer integration needs to be completed based on SkyRL API.")
    print("This script provides the foundation with LoRA and model loading.")


if __name__ == "__main__":
    main()

