"""
SFT (Supervised Fine-Tuning): gemma-2-2b on Reasoning Dataset
==============================================================
Fine-tune gemma-2-2b base model on nohurry/Opus-4.6-Reasoning-3000x-filtered
before performing Knowledge Distillation.

Pipeline: SFT (this script) → Distillation (distill_gemma.py)

Usage:
    python sft_gemma.py --config distill_config.yaml
    python sft_gemma.py --num_train_epochs 3 --learning_rate 2e-5
    python sft_gemma.py --max_steps 100  # quick test

After SFT, run distillation with the SFT checkpoint:
    python distill_gemma.py --student_model ./sft_output --config distill_config.yaml
"""

import argparse
import logging
import os
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# API Keys & Login
# ──────────────────────────────────────────────────────────────
MY_WANDB_KEY = "API_KEY"
MY_HF_TOKEN = "API_KEY"

try:
    if MY_WANDB_KEY:
        import wandb
        wandb.login(key=MY_WANDB_KEY)
        print("W&B logged in successfully!")
    else:
        print("W&B Key is empty. Skipping W&B login.")
except Exception as e:
    print(f"Failed to login to W&B: {e}")

try:
    if MY_HF_TOKEN:
        from huggingface_hub import login
        login(token=MY_HF_TOKEN)
        print("Hugging Face logged in successfully!")
    else:
        print("HF Token is empty. Skipping HF login.")
except Exception as e:
    print(f"Failed to login to Hugging Face: {e}")


# ──────────────────────────────────────────────────────────────
# Dataset Formatting
# ──────────────────────────────────────────────────────────────
def format_reasoning_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, str]:
    """
    Format Opus Reasoning dataset into chat-style training text.

    Dataset columns: id, problem, thinking, solution, difficulty, category, timestamp, hash
    """
    problem = example.get("problem", "").strip()
    if not problem:
        problem = example.get("question", "").strip()
    thinking = example.get("thinking", "").strip()
    solution = example.get("solution", "").strip()

    # Build the response with reasoning + answer
    response_parts = []
    if thinking:
        response_parts.append(f"<reasoning>\n{thinking}\n</reasoning>")
    if solution:
        response_parts.append(f"<answer>\n{solution}\n</answer>")

    response = "\n\n".join(response_parts) if response_parts else f"<answer>\n{solution}\n</answer>"

    messages = [
        {"role": "user", "content": problem},
        {"role": "model", "content": response} # Using role 'model' for Gemma API template format
    ]

    # Gemma chat format via template
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": text}


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
def load_sft_config(config_path: str, cli_args) -> Dict[str, Any]:
    """Load SFT config from YAML + CLI overrides."""
    defaults = {
        "student_model": "google/gemma-2-2b",
        "dataset_name": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "max_seq_length": 8192,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_steps": -1,
        "sft_output_dir": "./sft_output",
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 3,
        "report_to": "wandb",
        "push_to_hub": True,
        "hub_model_id": "Phonsiri/gemma-2-2b-SFT-Reasoning",
    }

    # Load YAML
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Only read SFT-safe keys from top-level (skip distillation-specific ones)
        skip_from_toplevel = {"hub_model_id", "push_to_hub", "output_dir", "dataset_name"}
        for key in defaults:
            if key in yaml_config and key not in skip_from_toplevel:
                defaults[key] = yaml_config[key]

        # SFT-specific section takes priority
        sft_section = yaml_config.get("sft", {})
        if sft_section:
            defaults.update(sft_section)

    # CLI overrides
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key == "config":
            continue
        if value is not None and key in defaults:
            defaults[key] = value

    return defaults


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SFT: gemma-2-2b on Reasoning Dataset")
    parser.add_argument("--config", type=str, default="distill_config.yaml")
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--sft_output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_sft_config(args.config, args)

    logger.info("=" * 60)
    logger.info("SFT Configuration (Phase 1: before Distillation)")
    logger.info("=" * 60)
    logger.info(f"  Model   : {config['student_model']}")
    logger.info(f"  Dataset : {config['dataset_name']}")
    logger.info(f"  MaxLen  : {config['max_seq_length']}")
    logger.info(f"  Mode    : Full Fine-Tuning (all parameters)")
    logger.info(f"  LR      : {config['learning_rate']}")
    logger.info(f"  Epochs  : {config['num_train_epochs']}")
    logger.info(f"  Output  : {config['sft_output_dir']}")
    logger.info("=" * 60)

    # --- Load tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"], trust_remote_code=True
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' or message['role'] == 'model' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- Load dataset ---
    dataset_name = config["dataset_name"]
    logger.info(f"Loading dataset: {dataset_name}")
    
    datasets_to_concat = []
    
    if isinstance(dataset_name, str):
        dataset_paths = [p.strip() for p in dataset_name.split(",") if p.strip()]
    elif isinstance(dataset_name, list):
        dataset_paths = dataset_name
    else:
        dataset_paths = [dataset_name]
        
    for path in dataset_paths:
        try:
            if path.endswith(".json") or path.endswith(".jsonl"):
                logger.info(f"Loading local JSON dataset: {path}")
                ds = load_dataset("json", data_files=path, split="train")
            else:
                logger.info(f"Loading HuggingFace dataset: {path}")
                ds = load_dataset(path, split="train")
            datasets_to_concat.append(ds)
        except Exception as e:
            logger.error(f"Failed to load dataset '{path}': {e}")
            raise e

    if not datasets_to_concat:
        raise ValueError("No datasets could be loaded.")
        
    if len(datasets_to_concat) > 1:
        logger.info(f"Concatenating {len(datasets_to_concat)} datasets...")
        
        # We need to make sure the columns match before concatenating.
        # Often HF datasets and local JSONs have different columns.
        # A simple approach is to keep only the necessary columns or select common ones.
        # But `sft_gemma.py` maps everything using `format_reasoning_example`, so as long as they
        # have `problem` or `question` and `solution` or `thinking`, it should be fine.
        # However, `concatenate_datasets` requires matching columns.
        # So we format EACH dataset first, then concatenate.
        
        formatted_datasets = []
        for ds in datasets_to_concat:
            original_columns = ds.column_names
            formatted_ds = ds.map(lambda x: format_reasoning_example(x, tokenizer), remove_columns=original_columns)
            formatted_datasets.append(formatted_ds)
            
        dataset = concatenate_datasets(formatted_datasets)
    else:
        dataset = datasets_to_concat[0]
        original_columns = dataset.column_names
        dataset = dataset.map(lambda x: format_reasoning_example(x, tokenizer), remove_columns=original_columns)
        
    logger.info(f"  Total formulated examples: {len(dataset)}")

    # Shuffle dataset
    logger.info("Shuffling dataset...")
    dataset = dataset.shuffle(seed=42)

    # Show a sample
    logger.info(f"\n--- Sample ---\n{dataset[0]['text'][:500]}...\n--------------")

    # --- Load model ---
    logger.info(f"Loading model: {config['student_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        device_map="auto",
        dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} (100% - Full Fine-Tuning)")

    # --- SFT Config ---
    sft_config = SFTConfig(
        output_dir=config["sft_output_dir"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        push_to_hub=config["push_to_hub"],
        hub_model_id=config["hub_model_id"] if config["hub_model_id"] else None,
        max_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=False,  # Requires flash_attention_2; disabled for sdpa
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # --- Create SFT Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --- Train ---
    logger.info("🚀 Starting SFT training...")
    train_result = trainer.train()

    # --- Save ---
    logger.info(f"Saving model to {config['sft_output_dir']}")
    trainer.save_model(config["sft_output_dir"])
    tokenizer.save_pretrained(config["sft_output_dir"])

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if config["push_to_hub"] and config["hub_model_id"]:
        logger.info(f"Pushing to Hub: {config['hub_model_id']}")
        trainer.push_to_hub()

    logger.info("✅ SFT complete!")
    logger.info(f"   Model saved to: {config['sft_output_dir']}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Next step: Run Distillation")
    logger.info("=" * 60)
    logger.info(f"  python distill_gemma.py \\")
    logger.info(f"      --student_model {config['sft_output_dir']} \\")
    logger.info(f"      --config distill_config.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
