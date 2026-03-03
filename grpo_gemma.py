"""
GRPO (Group Relative Policy Optimization): gemma-2-2b Reasoning
================================================================
Fine-tune gemma-2-2b (after SFT) using GRPO with Format and Accuracy rewards.
This script uses trl's GRPOTrainer.

Usage:
    python grpo_gemma.py --model_name_or_path ./sft_output --dataset_name AI-MO/NuminaMath-TIR
"""

import re
import os
import argparse
import logging
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Reward Functions
# -----------------------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
    """Extract final answer from <answer> tags if they exist. Otherwise return full text."""
    # Often models don't use <answer> but \boxed{}
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \boxed{}"""
    pattern = r'\\boxed{([^}]*)}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if the completion follows the <reasoning>...</reasoning> format.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        if "<reasoning>" in text and "</reasoning>" in text:
            # Basic validation: `<reasoning>` must appear before `</reasoning>`
            if text.find("<reasoning>") < text.find("</reasoning>"):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that checks if the final answer matches the ground truth.
    The `answer` argument comes from the dataset columns.
    """
    rewards = []
    for completion, gt_ans in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        
        # 1. Try to extract from \boxed{}
        pred = extract_boxed_answer(text)
        
        # 2. Try to exact ground truth from \boxed{} in the solution column if provided
        gt = extract_boxed_answer(gt_ans) if gt_ans else gt_ans.strip()
        
        # Super strict string matching (in practice you might want SymPy checking for math)
        if pred is not None and gt is not None and pred == gt:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
            
    return rewards


# -----------------------------------------------------------------------------
# Dataset Formatting
# -----------------------------------------------------------------------------

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process must be enclosed within <reasoning> and </reasoning> tags.
Finally, place the exact final answer inside \boxed{}.
"""

def prepare_dataset(dataset_name: str, split: str = "train"):
    """
    Prepare dataset for GRPO. We need 'prompt' (string or list of dicts) 
    and 'answer' (for the reward function).
    """
    dataset_names = [d.strip() for d in dataset_name.split(",")]
    all_datasets = []

    for path in dataset_names:
        try:
            if path.endswith(".json") or path.endswith(".jsonl"):
                print(f"Loading local JSON dataset for GRPO: {path}")
                ds = load_dataset("json", data_files=path, split="train")
            else:
                print(f"Loading HuggingFace dataset for GRPO: {path}")
                ds = load_dataset(path, split="train")
            all_datasets.append(ds)
        except Exception as e:
            print(f"Failed to load dataset '{path}': {e}")
            raise e

    from datasets import concatenate_datasets
    if not all_datasets:
        raise ValueError("No datasets could be loaded.")
        
    if len(all_datasets) > 1:
        print(f"Concatenating {len(all_datasets)} datasets...")
        dataset = concatenate_datasets(all_datasets)
    else:
        dataset = all_datasets[0]
    def format_row(example):
        # Specific formatting based on known math datasets
        if "problem" in example and "solution" in example:
            prompt = example["problem"]
            solution = example["solution"]
        elif "question" in example and "answer" in example:
            prompt = example["question"]
            solution = example["answer"]
        else:
            prompt = example.get("text", "")
            solution = ""
            
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
        ]
        
        return {
            "prompt": messages,
            "answer": solution  # Keep solution to check against in accuracy reward
        }
        
    dataset = dataset.map(format_row)
    return dataset


# -----------------------------------------------------------------------------
# Main Training Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Phonsiri/gemma-2-2b-Distillation-gemma-2-27b-it", help="Path to Starting model")
    parser.add_argument("--dataset_name", type=str, default="open-r1/OpenR1-Math-220k", help="Hugging Face dataset name or local path")
    parser.add_argument("--output_dir", type=str, default="./grpo_output", help="Where to save the model")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Increased for Reasoning")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions to generate per prompt (G)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Smaller LR for RL")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--hub_model_id", type=str, default="Phonsiri/gemma-2-2b-GRPO-Reasoning-full")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint folder to resume from")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Initializing GRPO Training")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info("==================================================")

    # 1. Load Dataset
    logger.info("Loading and formatting dataset...")
    train_dataset = prepare_dataset(args.dataset_name)
    logger.info(f"Train dataset size: {len(train_dataset)}")

    # 2. Configure GRPO
    # Note: trl's GRPOConfig combines standard TrainingArguments with RL-specific args
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=10,            # Changed from 100 to 10
        save_total_limit=3,
        report_to="wandb",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="checkpoint", # Added: Push to hub at every save_steps
        
        # Generation Specific
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=0.9,
        
        # GRPO specific
        beta=0.01,  # KL penalty coefficient
    )

    # 3. Load Model and Tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # For standard GRPO, we load the policy model (trainable)
    # The reference model is automatically created by GRPOTrainer unless specified
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Initialize Trainer
    logger.info("Setting up GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 5. Train
    logger.info("🚀 Starting GRPO Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 6. Save
    logger.info(f"Saving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.push_to_hub:
        logger.info(f"Pushing to HF Hub: {args.hub_model_id}")
        trainer.push_to_hub()
        
    logger.info("✅ GRPO Training Complete!")

if __name__ == "__main__":
    main()
