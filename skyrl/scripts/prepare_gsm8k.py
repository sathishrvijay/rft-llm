#!/usr/bin/env python3
"""
Prepare GSM8K dataset for SkyRL training.
Following SkyRL examples for dataset preparation.
"""

import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List


def prepare_gsm8k_dataset(
    output_dir: str,
    split: str = "main",
    train_split: float = 0.9,
    format_type: str = "skyrl"
) -> None:
    """
    Prepare GSM8K dataset for SkyRL training.
    
    Args:
        output_dir: Directory to save prepared dataset
        split: Dataset split to use
        train_split: Fraction of data to use for training
        format_type: Format type for SkyRL pipeline
    """
    print("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", split=split)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=1.0 - train_split, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Format dataset for SkyRL
    def format_example(example: Dict) -> Dict:
        """Format a single example for SkyRL."""
        question = example["question"]
        answer = example["answer"]
        
        # Extract final answer (GSM8K format: "### Answer: 42")
        final_answer = answer.split("###")[-1].strip()
        if final_answer.startswith("Answer:"):
            final_answer = final_answer.replace("Answer:", "").strip()
        
        return {
            "question": question,
            "answer": answer,
            "final_answer": final_answer,
            "prompt": f"Solve the following math problem step by step.\n\nProblem: {question}\n\nSolution:",
        }
    
    # Process datasets
    print("Formatting datasets...")
    train_formatted = [format_example(ex) for ex in train_dataset]
    eval_formatted = [format_example(ex) for ex in eval_dataset]
    
    # Save datasets
    train_file = output_path / "train.jsonl"
    eval_file = output_path / "eval.jsonl"
    
    print(f"Saving train dataset to {train_file}...")
    with open(train_file, "w") as f:
        for ex in train_formatted:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Saving eval dataset to {eval_file}...")
    with open(eval_file, "w") as f:
        for ex in eval_formatted:
            f.write(json.dumps(ex) + "\n")
    
    # Save metadata
    metadata = {
        "dataset": "gsm8k",
        "train_size": len(train_formatted),
        "eval_size": len(eval_formatted),
        "format": format_type,
        "train_file": str(train_file),
        "eval_file": str(eval_file),
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Train file: {train_file}")
    print(f"Eval file: {eval_file}")
    print(f"Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset for SkyRL")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/gsm8k",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="main",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="skyrl",
        help="Format type for SkyRL pipeline"
    )
    
    args = parser.parse_args()
    
    prepare_gsm8k_dataset(
        output_dir=args.output_dir,
        split=args.split,
        train_split=args.train_split,
        format_type=args.format
    )


if __name__ == "__main__":
    main()

