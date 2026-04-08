"""
eval_sft.py — Evaluate SFT checkpoints with live search.
Supports resuming from interrupted runs.

Usage:
    CUDA_VISIBLE_DEVICES=3 python eval_sft.py \
        --adapter ./ckpt_a/final_adapter \
        --questions eval_questions_500.jsonl \
        --output eval_output/ckpt_a_trajectories.jsonl

    # If interrupted, just re-run the same command — it picks up where it left off.

    python check_success.py --path eval_output/ckpt_a_trajectories.jsonl
"""

import argparse
import json
import re
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B"
RETRIEVER_URL = "http://127.0.0.1:8001/retrieve"
TOPK = 3
MAX_NEW_TOKENS = 500
MAX_TURNS = 5
MAX_SEQ_LENGTH = 4096


# ---------------------------------------------------------------------------
# Search function
# ---------------------------------------------------------------------------

def search(query, retriever_url=RETRIEVER_URL, topk=TOPK):
    try:
        payload = {"queries": [query], "topk": topk}
        response = requests.post(retriever_url, json=payload, timeout=30)
        results = response.json()["result"][0]
        formatted = []
        for i, doc in enumerate(results):
            title = doc.get("title", "")
            contents = doc.get("contents", "")
            if title:
                formatted.append(f"Doc {i+1}(Title: \"{title}\") {contents}")
            else:
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
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    )

    print(f"Loading adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.to(device)
    model.eval()

    # Load questions from JSONL
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
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    if completed >= len(records):
        print("All questions already completed!")
        return

    # Check retriever
    print(f"Testing retriever at {RETRIEVER_URL}...")
    try:
        search("test query")
        print(f"  Retriever OK")
    except Exception as e:
        print(f"  ERROR: Retriever not reachable: {e}")
        return

    # Run evaluation, appending one trajectory at a time
    # Open in append mode so we don't overwrite completed results
    mode = "a" if completed > 0 else "w"
    with open(args.output, mode, encoding="utf-8") as out_f:
        for idx in range(completed, len(records)):
            rec = records[idx]

            if idx % 10 == 0:
                print(f"Processing {idx}/{len(records)}...")

            prompt = rec["prompt"] + "\n"
            ground_truth = rec.get("ground_truth", "")

            steps, full_response = generate_with_search(model, tokenizer, prompt, device)

            traj = {
                "question": prompt,
                "ground_truth": ground_truth,
                "steps": steps,
            }

            # Write immediately and flush — saved even if we crash
            out_f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone! All {len(records)} trajectories saved to {args.output}")
    print(f"Run evaluation with:")
    print(f"  python check_success.py --path {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
