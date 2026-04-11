"""
eval_grpo.py — Evaluate GRPO checkpoints with live search.
Supports resuming from interrupted runs.

Usage:
    CUDA_VISIBLE_DEVICES=3 python eval_grpo.py \
        --checkpoint ./verl_checkpoints/grpo-adapter-a/actor/global_step_100 \
        --questions data/grpo/test.parquet \
        --output eval_output/grpo_a_trajectories.jsonl

    python check_success.py --path eval_output/grpo_a_trajectories.jsonl
"""

import argparse
import json
import re
import os
import torch
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
TOPK = 3
MAX_NEW_TOKENS = 500
MAX_TURNS = 5
MAX_SEQ_LENGTH = 4096


# ---------------------------------------------------------------------------
# Search function
# ---------------------------------------------------------------------------

def search(query, retriever_url=RETRIEVER_URL, topk=TOPK):
    try:
        payload = {"queries": [query], "topk": topk, "return_scores": True}
        response = requests.post(retriever_url, json=payload, timeout=30)
        results = response.json()["result"][0]
        formatted = []
        for i, item in enumerate(results):
            contents = item["document"].get("contents", "")
            formatted.append(f"Doc {i+1}: {contents}")
        return " ".join(formatted)
    except Exception as e:
        print(f"  [WARNING] Search failed: {e}")
        return "No results found."
# ---------------------------------------------------------------------------
# Generation with search loop
# ---------------------------------------------------------------------------

def generate_with_search(model, tokenizer, prompt, device):
    steps = []
    full_response = ""
    current_input = prompt

    for turn in range(MAX_TURNS):
        inputs = tokenizer(
            current_input,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_response += generated_text

        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", generated_text, re.DOTALL)
        search_match = re.search(r"<search>\s*(.*?)\s*</search>", generated_text, re.DOTALL)
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", generated_text, re.DOTALL)

        if think_match:
            steps.append({
                "action": "think",
                "content": think_match.group(1).strip()
            })

        if search_match and not answer_match:
            query = search_match.group(1).strip()
            search_results = search(query)
            steps.append({
                "action": "search",
                "query": query,
                "search_results": search_results,
            })
            info_block = f" <information> {search_results} </information> "
            current_input = current_input + generated_text + info_block
            full_response += info_block
            continue

        if answer_match:
            steps.append({
                "action": "answer",
                "content": answer_match.group(1).strip()
            })
            break
        else:
            steps.append({
                "action": "answer",
                "content": generated_text.strip()
            })
            break

    return steps, full_response


# ---------------------------------------------------------------------------
# Load questions from parquet
# ---------------------------------------------------------------------------

def load_questions_from_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"][0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]["target"][0]
        records.append({
            "prompt": prompt,
            "ground_truth": ground_truth
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading GRPO checkpoint from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")

    # Load questions
    if args.questions.endswith(".parquet"):
        records = load_questions_from_parquet(args.questions)
    else:
        records = []
        with open(args.questions, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    print(f"Loaded {len(records)} questions")

    # Check how many are already completed (for resume)
    completed = 0
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                if line.strip():
                    completed += 1
        print(f"Resuming from question {completed}/{len(records)}")
    else:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    if completed >= len(records):
        print("All questions already completed!")
        return

    # Test retriever
    print(f"Testing retriever at {RETRIEVER_URL}...")
    try:
        search("test query")
        print(f"  Retriever OK")
    except Exception as e:
        print(f"  ERROR: Retriever not reachable: {e}")
        return

    # Run evaluation
    mode = "a" if completed > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        for idx in range(completed, len(records)):
            rec = records[idx]

            if idx % 10 == 0:
                print(f"Processing {idx}/{len(records)}...")

            prompt = rec["prompt"]
            ground_truth = rec.get("ground_truth", "")

            steps, full_response = generate_with_search(model, tokenizer, prompt, device)

            traj = {
                "question": prompt,
                "ground_truth": ground_truth,
                "steps": steps,
            }

            out_f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone! Saved to {args.output}")
    print(f"Run: python check_success.py --path {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to verl checkpoint folder")
    parser.add_argument("--questions", required=True, help="Path to test parquet or jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    args = parser.parse_args()
    main(args)
