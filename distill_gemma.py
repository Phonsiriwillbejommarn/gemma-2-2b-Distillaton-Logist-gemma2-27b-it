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
    dataset_name: str = "tatsu-lab/alpaca"
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
def format_alpaca(example: Dict[str, Any]) -> Dict[str, str]:
    """Format Alpaca dataset into instruction-following format."""
    if example.get("input", "").strip():
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def format_math(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format rasbt/math_full_minus_math500 dataset.
    Columns: problem, level, type, solution, answer, unique_id

    Output: Qwen ChatML format with problem → solution.
    """
    problem = example.get("problem", "").strip()
    solution = example.get("solution", "").strip()
    answer = example.get("answer", "").strip()

    # Build response: solution with final answer
    response = solution
    if answer and answer not in solution:
        response += f"\n\nThe answer is $\\boxed{{{answer}}}$."

    # Gemma chat format
    text = (
        f"<start_of_turn>user\n{problem}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>"
    )

    return {"text": text}


def format_prompt_completion(example: Dict[str, Any]) -> Dict[str, str]:
    """Format prompt/completion datasets (e.g., CodeAlpaca) into ChatML."""
    text = (
        f"<start_of_turn>user\n{example.get('prompt', '').strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n{example.get('completion', '').strip()}<end_of_turn>"
    )
    return {"text": text}


def format_messages(example: Dict[str, Any]) -> Dict[str, str]:
    """Format standard conversational messages (e.g., Aurora-Alpha) into ChatML."""
    messages = example.get("messages", [])
    
    text = ""
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            continue
        if role == "assistant":
            role = "model"
        content = msg.get("content", "").strip()
        text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    return {"text": text.strip()}


def prepare_dataset(config: DistillConfig, tokenizer: AutoTokenizer):
    """Load, format, and tokenize single or multiple datasets."""
    dataset_names = [d.strip() for d in config.dataset_name.split(",")]
    all_datasets = []

    for name in dataset_names:
        logger.info(f"Loading dataset: {name}")
        if os.path.isfile(name) or os.path.isdir(name):
            if name.endswith(".jsonl"):
                ds = load_dataset("json", data_files=name, split="train")
            else:
                ds = load_dataset(name, split="train")
        else:
            ds = load_dataset(name, split="train")

        # Auto-detect format and apply formatting
        if "messages" in ds.column_names:
            logger.info(f"[{name}] Detected Messages format → formatting as ChatML...")
            ds = ds.map(format_messages, remove_columns=ds.column_names)
        elif "problem" in ds.column_names and "solution" in ds.column_names:
            logger.info(f"[{name}] Detected MATH format → formatting as ChatML...")
            ds = ds.map(format_math, remove_columns=ds.column_names)
        elif "prompt" in ds.column_names and "completion" in ds.column_names:
            logger.info(f"[{name}] Detected Prompt/Completion format → formatting as ChatML...")
            ds = ds.map(format_prompt_completion, remove_columns=ds.column_names)
        elif "instruction" in ds.column_names:
            logger.info(f"[{name}] Detected Alpaca format → formatting as ChatML...")
            ds = ds.map(format_alpaca, remove_columns=ds.column_names)
        elif config.dataset_text_field in ds.column_names:
            logger.info(f"[{name}] Using existing text column: {config.dataset_text_field}")
        else:
            logger.warning(f"[{name}] Unknown format with columns: {ds.column_names}. Hoping for best!")

        all_datasets.append(ds)

    if len(all_datasets) > 1:
        logger.info(f"Concatenating {len(all_datasets)} datasets...")
        dataset = concatenate_datasets(all_datasets)
    else:
        dataset = all_datasets[0]

    logger.info("Shuffling the final dataset...")
    dataset = dataset.shuffle(seed=42)

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

        # --- KL Divergence on softened distributions (memory efficient) ---
        T = self.temperature

        # Filter to only non-padding tokens BEFORE computing KL (saves ~19GB VRAM)
        labels = inputs.get("labels", None)
        if labels is not None:
            mask = (labels != -100)  # (B, seq_len)
            mask_flat = mask.view(-1)  # (B * seq_len)
            student_flat = student_logits_trimmed.view(-1, min_vocab)[mask_flat]  # (N, vocab)
            teacher_flat = teacher_logits_trimmed.view(-1, min_vocab)[mask_flat]  # (N, vocab)
        else:
            student_flat = student_logits_trimmed.view(-1, min_vocab)
            teacher_flat = teacher_logits_trimmed.view(-1, min_vocab)

        # Process in chunks to prevent OOM on massive batch/seq lengths (e.g. 4x8192)
        chunk_size = 512
        total_tokens = student_flat.size(0)
        kl_loss_sum = 0.0

        for i in range(0, total_tokens, chunk_size):
            s_chunk = student_flat[i : i + chunk_size]
            t_chunk = teacher_flat[i : i + chunk_size]

            s_log_probs = F.log_softmax(s_chunk / T, dim=-1)
            t_probs = F.softmax(t_chunk / T, dim=-1)

            # reduction="batchmean" divides by batch size (which is chunk size here)
            # We use "sum" and divide by total_tokens later to get exact mathematical equivalence
            chunk_kl = F.kl_div(s_log_probs, t_probs, reduction="sum")
            kl_loss_sum += chunk_kl

        kl_loss = kl_loss_sum / max(total_tokens, 1)

        # Scale by T^2 (standard distillation scaling)
        kl_loss = kl_loss * (T ** 2)

        # --- Combined loss ---
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}: "
                f"total={loss.item():.4f}  "
                f"kl={kl_loss.item():.4f}  "
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
