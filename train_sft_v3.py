"""
train_sft_v3.py — LoRA SFT training script (final version).

Key design choices to avoid NaN:
- bf16 base model + float32 LoRA params
- Manual cross-entropy in float32 (bf16 softmax over 152K vocab overflows)
- batch_size=1 to fit float32 logits in memory without gradient checkpointing
- grad_accum=32 for effective batch size of 32

Usage:
    CUDA_VISIBLE_DEVICES=3 python train_sft_v3.py \
        --data sft_checkpoint_a.jsonl --output ./ckpt_a
"""

import argparse
import json
import math
import os
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_SEQ_LENGTH = 4096

# Training hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 32       # effective batch = 1 * 32 = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
LOG_EVERY = 10
MIN_UNMASKED_TOKENS = 20


# ---------------------------------------------------------------------------
# Data loading + tokenization
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_tokenize_fn(tokenizer):
    info_open_ids = tokenizer("<information>", add_special_tokens=False).input_ids
    info_close_ids = tokenizer("</information>", add_special_tokens=False).input_ids

    def find_spans(ids, open_ids, close_ids):
        spans = []
        i = 0
        n_open, n_close = len(open_ids), len(close_ids)
        while i < len(ids):
            if ids[i:i + n_open] == open_ids:
                start = i
                j = i + n_open
                while j < len(ids):
                    if ids[j:j + n_close] == close_ids:
                        spans.append((start, j + n_close))
                        i = j + n_close
                        break
                    j += 1
                else:
                    i += 1
            else:
                i += 1
        return spans

    def tokenize_fn(example):
        full_text = example["prompt"] + example["response"]
        tokenized = tokenizer(
            full_text, max_length=MAX_SEQ_LENGTH,
            truncation=True, padding=False, return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        prompt_tokenized = tokenizer(
            example["prompt"], max_length=MAX_SEQ_LENGTH,
            truncation=True, padding=False, return_tensors=None,
        )
        prompt_len = len(prompt_tokenized["input_ids"])

        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        for start, end in find_spans(input_ids, info_open_ids, info_close_ids):
            for i in range(start, min(end, len(labels))):
                labels[i] = -100

        tokenized["labels"] = labels
        return tokenized

    return tokenize_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model + tokenizer ---
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.to(device)

    # --- Dataset ---
    print(f"Loading dataset from: {args.data}")
    records = load_jsonl(args.data)
    ds = Dataset.from_list(records)

    tokenize_fn = build_tokenize_fn(tokenizer)
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, desc="Tokenizing")

    pre_filter = len(ds)
    ds = ds.filter(
        lambda ex: sum(1 for l in ex["labels"] if l != -100) >= MIN_UNMASKED_TOKENS,
        desc="Filtering degenerate examples",
    )
    print(f"Dataset: {pre_filter} -> {len(ds)} after filtering")

    # --- DataLoader (batch_size=1, no padding needed) ---
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
    dataloader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=0,
    )

    # --- Optimizer + scheduler ---
    steps_per_epoch = math.ceil(len(dataloader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss function — computed in float32 to prevent bf16 softmax overflow
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    # --- Training loop ---
    global_step = 0
    model.train()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        micro_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass — no labels, we compute loss manually in float32
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # Cast logits to float32 before cross-entropy
            logits = outputs.logits.float()
            labels = batch["labels"]

            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            (loss / GRAD_ACCUM_STEPS).backward()

            running_loss += loss.item()
            micro_steps += 1

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                if global_step % LOG_EVERY == 0:
                    avg_loss = running_loss / micro_steps
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  step {global_step}/{total_steps} | "
                        f"epoch {epoch+1}/{NUM_EPOCHS} | "
                        f"loss {avg_loss:.4f} | "
                        f"grad_norm {grad_norm:.4f} | "
                        f"lr {lr:.6f}"
                    )
                    running_loss = 0.0
                    micro_steps = 0

        # Handle leftover micro-batches
        if micro_steps > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Save checkpoint at end of each epoch
        ckpt_path = os.path.join(args.output, f"epoch_{epoch+1}")
        print(f"  Saving checkpoint: {ckpt_path}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    # --- Save final adapter ---
    adapter_save_path = os.path.join(args.output, "final_adapter")
    print(f"Saving final LoRA adapter to: {adapter_save_path}")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args)
