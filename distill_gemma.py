"""
Knowledge Distillation: gemma-3-27b-it → gemma-2-2b
===========================================================
Logit-based distillation with KL-divergence + Cross-Entropy loss.

Usage:
    python distill_gemma.py --config distill_config.yaml
    python distill_gemma.py --alpha 0.7 --temperature 3.0 --num_train_epochs 5

The teacher model is loaded in 4-bit quantization (frozen).
The student model is trained with optional LoRA adapters.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

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

# --- Login Hugging Face ---
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
# Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class DistillConfig:
    """All configuration for the distillation run."""

    # Models
    teacher_model: str = "google/gemma-3-27b-it"
    student_model: str = "google/gemma-2-2b"

    # Dataset
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    max_samples: int = 20000 
    dataset_text_field: str = "text"
    max_seq_length: int = 2048

    # Distillation
    alpha: float = 0.5       # weight for KL-div loss
    temperature: float = 2.0  # softmax temperature

    # Training
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_steps: int = -1  # override epochs if > 0

    # Teacher quantization
    teacher_load_in_4bit: bool = True
    teacher_bnb_4bit_compute_dtype: str = "bfloat16"
    teacher_bnb_4bit_quant_type: str = "nf4"

    # Output
    output_dir: str = "./distill_output"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 5
    report_to: str = "wandb"
    push_to_hub: bool = False
    hub_model_id: str = ""


def load_config(config_path: Optional[str] = None, cli_overrides: Optional[Dict] = None) -> DistillConfig:
    """Load config from YAML file and apply CLI overrides."""
    config = DistillConfig()

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            for key, value in yaml_config.items():
                if key == "generation":
                    continue  # skip generation section (used by other script)
                if hasattr(config, key):
                    setattr(config, key, value)

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

    return config


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
def format_reasoning_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, str]:
    """
    Format math/reasoning dataset into chat-style training text.
    """
    problem = example.get("problem", "").strip()
    if not problem:
        problem = example.get("question", "").strip()
        
    # Inject a system-like instruction to force the teacher to expect the tags, directly lowering KL divergence
    problem += "\n\nPlease reason step by step, and put your thoughts within <reasoning> and </reasoning> tags, and your final answer within <answer> and </answer> tags."
        
    thinking = example.get("thinking", "").strip()
    
    # Support for OpenR1-Math-220k which stores reasoning inside 'generations'
    if not thinking and "generations" in example and example["generations"]:
        gen = example["generations"][0]
        import re
        think_match = re.search(r"<think>(.*?)</think>", gen, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            
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
        {"role": "model", "content": response}
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    except ValueError:
        text = ""
    return {"text": text}


def prepare_dataset(config: DistillConfig, tokenizer: AutoTokenizer):
    """Load, format, and tokenize single or multiple datasets."""
    dataset_names = [d.strip() for d in config.dataset_name.split(",")]
    all_datasets = []

    for path in dataset_names:
        try:
            if path.endswith(".json") or path.endswith(".jsonl"):
                logger.info(f"Loading local JSON dataset: {path}")
                ds = load_dataset("json", data_files=path, split="train")
            else:
                logger.info(f"Loading HuggingFace dataset: {path}")
                ds = load_dataset(path, split="train")
            all_datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to load dataset '{path}': {e}")
            raise e

    if not all_datasets:
        raise ValueError("No datasets could be loaded.")

    if len(all_datasets) > 1:
        logger.info(f"Concatenating {len(all_datasets)} datasets...")
        
        formatted_datasets = []
        for ds in all_datasets:
            original_columns = ds.column_names
            formatted_ds = ds.map(lambda x: format_reasoning_example(x, tokenizer), remove_columns=original_columns)
            formatted_datasets.append(formatted_ds)
            
        dataset = concatenate_datasets(formatted_datasets)
    else:
        dataset = all_datasets[0]
        original_columns = dataset.column_names
        dataset = dataset.map(lambda x: format_reasoning_example(x, tokenizer), remove_columns=original_columns)

    logger.info("Shuffling the final dataset...")
    dataset = dataset.shuffle(seed=42)

    if hasattr(config, "max_samples") and config.max_samples > 0 and len(dataset) > config.max_samples:
        logger.info(f"Limiting dataset to {config.max_samples} randomly selected examples (from {len(dataset)})...")
        dataset = dataset.select(range(config.max_samples))

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples[config.dataset_text_field],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Filter out examples that were truncated (length > max_seq_length initially)
    # We check if it ends without padding, or if we want to be safe, just filter before padding.
    # Actually, a better way is to tokenize without padding/truncation first, filter by length, then pad.
    def filter_long_examples(example):
        # We can check token length
        tokens = tokenizer(example[config.dataset_text_field], truncation=False)
        return len(tokens["input_ids"]) <= config.max_seq_length
        
    logger.info("Filtering out examples that exceed maximum sequence length...")
    original_size = len(dataset)
    dataset = dataset.filter(filter_long_examples)
    filtered_size = len(dataset)
    logger.info(f"Filtered out {original_size - filtered_size} examples. Remaining: {filtered_size}")

    logger.info("Re-tokenizing with padding arrays...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.with_format("torch")

    # Create labels (same as input_ids for causal LM, pad tokens → -100)
    def add_labels(examples):
        labels = examples["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        examples["labels"] = labels
        return examples

    tokenized = tokenized.map(add_labels, batched=True)

    logger.info(f"Dataset ready: {len(tokenized)} examples")
    return tokenized


# ──────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────
def load_teacher(config: DistillConfig) -> AutoModelForCausalLM:
    """Load the teacher model in 4-bit quantization (frozen)."""
    logger.info(f"Loading teacher: {config.teacher_model}")

    if config.teacher_load_in_4bit:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        compute_dtype = dtype_map.get(config.teacher_bnb_4bit_compute_dtype, torch.bfloat16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.teacher_bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=True,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Freeze all teacher parameters
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logger.info(f"Teacher loaded ({sum(p.numel() for p in teacher.parameters()) / 1e9:.1f}B params, frozen)")
    return teacher


def load_student(config: DistillConfig) -> AutoModelForCausalLM:
    """Load the student model for full fine-tuning."""
    logger.info(f"Loading student: {config.student_model}")

    student = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        device_map="auto",
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student loaded: {trainable_params:,} / {total_params:,} params (100% - Full Fine-Tuning)")

    return student


# ──────────────────────────────────────────────────────────────
# Distillation Trainer
# ──────────────────────────────────────────────────────────────
class DistillationTrainer(Trainer):
    """
    Custom Trainer that computes the distillation loss:
        L = α * KL(softened_teacher || softened_student) + (1-α) * CE(labels, student)
    """

    def __init__(self, teacher_model: AutoModelForCausalLM, alpha: float,
                 temperature: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        logger.info(f"DistillationTrainer: α={alpha}, T={temperature}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute combined distillation + cross-entropy loss."""
        # --- Student forward ---
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits  # (B, seq_len, vocab)
        ce_loss = student_outputs.loss            # standard CE loss from labels

        # --- Teacher forward (no grad) ---
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            teacher_logits = teacher_outputs.logits  # (B, seq_len, vocab)

        # --- Align vocab sizes if different ---
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits_trimmed = student_logits[..., :min_vocab]
        teacher_logits_trimmed = teacher_logits[..., :min_vocab]

        # --- Cross-Entropy KD Loss (auto-handles vocab mismatch via soft labels) ---
        T = self.temperature

        # Filter to only non-padding tokens BEFORE computing loss (saves ~19GB VRAM)
        labels = inputs.get("labels", None)
        if labels is not None:
            mask = (labels != -100)  # (B, seq_len)
            mask_flat = mask.view(-1)  # (B * seq_len)
            student_flat = student_logits_trimmed.view(-1, min_vocab)[mask_flat]  # (N, vocab)
            teacher_flat = teacher_logits_trimmed.view(-1, min_vocab)[mask_flat]  # (N, vocab)
        else:
            student_flat = student_logits_trimmed.view(-1, min_vocab)
            teacher_flat = teacher_logits_trimmed.view(-1, min_vocab)

        # Process in chunks to prevent OOM on massive batch/seq lengths
        # REDUCED chunk size specifically to handle extreme sequence lengths (>8192)
        chunk_size = 64
        total_tokens = student_flat.size(0)
        distill_loss_sum = 0.0

        for i in range(0, total_tokens, chunk_size):
            s_chunk = student_flat[i : i + chunk_size]
            t_chunk = teacher_flat[i : i + chunk_size]

            # Cross-Entropy pseudo-code for KD: F.cross_entropy(student_logits, soft_labels)
            # We scale logits by T to match standard temp softening
            soft_labels = F.softmax(t_chunk / T, dim=-1)
            chunk_distill = F.cross_entropy(s_chunk / T, soft_labels, reduction="sum")
            distill_loss_sum += chunk_distill

        distill_loss = distill_loss_sum / max(total_tokens, 1)

        # Scale by T^2 (standard distillation scaling)
        distill_loss = distill_loss * (T ** 2)

        # --- Combined loss ---
        loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}: "
                f"total={loss.item():.4f}  "
                f"kl={distill_loss.item():.4f}  "
                f"ce={ce_loss.item():.4f}"
            )

        return (loss, student_outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation: gemma-3-27b-it → gemma-2-2b")
    parser.add_argument("--config", type=str, default="distill_config.yaml", help="YAML config file")

    # Allow CLI overrides for common params
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir, or 'auto' to find latest in output_dir")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build overrides dict from CLI
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "resume_from_checkpoint")}

    config = load_config(args.config, overrides)

    # Log config summary
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Configuration")
    logger.info("=" * 60)
    logger.info(f"  Teacher : {config.teacher_model}")
    logger.info(f"  Student : {config.student_model}")
    logger.info(f"  Dataset : {config.dataset_name}")
    logger.info(f"  Alpha   : {config.alpha}")
    logger.info(f"  Temp    : {config.temperature}")
    logger.info(f"  Mode    : Full Fine-Tuning (all parameters)")
    logger.info(f"  LR      : {config.learning_rate}")
    logger.info(f"  Epochs  : {config.num_train_epochs}")
    logger.info(f"  Output  : {config.output_dir}")
    logger.info("=" * 60)

    # --- Load tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add fallback chat template if not present (as in SFT)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% elif message['role'] == 'assistant' or message['role'] == 'model' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"""

    tokenizer.padding_side = "right"
    # --- Load dataset ---
    train_dataset = prepare_dataset(config, tokenizer)

    # --- Load models ---
    teacher = load_teacher(config)
    student = load_student(config)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.hub_model_id else None,
        hub_strategy="checkpoint",  # Push every checkpoint to HF Hub
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # --- Create Trainer ---
    trainer = DistillationTrainer(
        teacher_model=teacher,
        alpha=config.alpha,
        temperature=config.temperature,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # --- Train (with checkpoint resume support) ---
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            # Auto-find latest checkpoint in output_dir
            checkpoints = [
                os.path.join(config.output_dir, d)
                for d in os.listdir(config.output_dir)
                if d.startswith("checkpoint-") or d == "last-checkpoint"
            ] if os.path.isdir(config.output_dir) else []
            if checkpoints:
                resume_checkpoint = max(checkpoints, key=os.path.getmtime)
                logger.info(f"🔄 Auto-resuming from latest checkpoint: {resume_checkpoint}")
            else:
                logger.info("No checkpoint found, starting from scratch.")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            logger.info(f"🔄 Resuming from checkpoint: {resume_checkpoint}")

    logger.info("Starting distillation training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Save ---
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if config.push_to_hub and config.hub_model_id:
        logger.info(f"Pushing to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub()

    logger.info("✅ Distillation complete!")
    logger.info(f"   Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
