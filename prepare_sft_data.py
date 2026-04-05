"""
prepare_sft_data.py

Converts trajectory JSONL files into SFT training data for LoRA fine-tuning.

Produces two output files:
  - sft_checkpoint_a.jsonl : success trajectories only (Checkpoint A)
  - sft_checkpoint_b.jsonl : success + STaR-rationalised failed trajectories (Checkpoint B)

Each output line is a JSON object with two fields:
  - "prompt"    : the full instruction prompt (taken directly from the trajectory's "question" field)
  - "response"  : the reconstructed <think>/<search>/<information>/<answer> sequence

Usage:
  python prepare_sft_data.py \
    --success success_traj.jsonl \
    --failed  failed_traj.jsonl \
    --out_a   sft_checkpoint_a.jsonl \
    --out_b   sft_checkpoint_b.jsonl
"""

import json
import argparse


# ---------------------------------------------------------------------------
# Core reconstruction logic
# ---------------------------------------------------------------------------

def reconstruct_response(steps, ground_truth=None):
    """
    Converts a list of trajectory steps into a single response string
    using the Search-R1 token format.

    For each step:
      - "think"  -> <think> content </think>
      - "search" -> <search> query </search><information> search_results </information>
      - "answer" -> <answer> content </answer>

    The `ground_truth` argument is only used for STaR rationalisation (Checkpoint B).
    When provided, the content of the final "answer" step is replaced with the
    ground truth string, correcting the wrong answer from a failed trajectory.

    Why does <information> appear in the response string at all?
    Because SFT trains by teacher-forcing over the full sequence — the model
    sees the entire trajectory at once during training. The <information> tokens
    are included so the model learns the correct positional context (i.e. that
    retrieved results appear between a search call and the next think step).
    However, they must be masked out during loss computation so the model is
    not penalised for or rewarded by the retrieved content it did not generate.
    This masking is handled separately in the LoRA training script.
    """
    parts = []

    for i, step in enumerate(steps):
        action = step["action"]

        if action == "think":
            parts.append(f"<think> {step['content']} </think>")

        elif action == "search":
            parts.append(f"<search> {step['query']} </search>")
            parts.append(f"<information> {step['search_results']} </information>")

        elif action == "answer":
            # For failed trajectories (Checkpoint B), override the wrong answer
            # with the ground truth. This is the STaR rationalisation step:
            # we are constructing a "what should have been said" target.
            answer_content = ground_truth if ground_truth is not None else step["content"]
            parts.append(f"<answer> {answer_content} </answer>")

    return " ".join(parts)


def process_trajectory(record, star_rationalize=False):
    """
    Converts one JSONL record into a (prompt, response) pair.

    The "question" field already contains the full prompt including the
    system message, instruction template, and the actual question — exactly
    as it was fed to the model during rollout. We reuse it verbatim.

    If star_rationalize=True, the ground truth is injected into the answer
    step. This is only set to True when processing failed trajectories for
    Checkpoint B.
    """
    prompt = record["question"]
    ground_truth = record["ground_truth"] if star_rationalize else None
    response = reconstruct_response(record["steps"], ground_truth=ground_truth)

    return {"prompt": prompt, "response": response}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Reads a JSONL file and yields one parsed dict per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records, path):
    """Writes a list of dicts to a JSONL file, one per line."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):

    # --- Checkpoint A: success trajectories only ---
    # The model learns to imitate reasoning chains that already produced
    # the correct answer. This is standard RFT (Rejection Fine-Tuning).
    print("Processing success trajectories (Checkpoint A)...")
    ckpt_a = []
    for record in read_jsonl(args.success):
        ckpt_a.append(process_trajectory(record, star_rationalize=False))

    write_jsonl(ckpt_a, args.out_a)

    # --- Checkpoint B: success + STaR-rationalised failed trajectories ---
    # We start from Checkpoint A's data, then augment with failed trajectories
    # that have been corrected via STaR rationalisation.
    #
    # Why include the failed trajectories at all?
    # The model attempted these questions but got them wrong, which means
    # it already produced a plausible reasoning structure — it just arrived
    # at the wrong answer. By injecting the ground truth into the answer step,
    # we create a training signal that says: "given this reasoning chain,
    # the correct conclusion is X." This teaches the model to connect
    # its own reasoning patterns to correct outputs, which is the core
    # insight of the STaR paper (Zelikman et al., 2022).
    print("Processing failed trajectories with STaR rationalisation (Checkpoint B)...")
    ckpt_b = list(ckpt_a)  # start with all of Checkpoint A's data
    skipped = 0

    for record in read_jsonl(args.failed):
        # Safety check: skip records missing required fields
        if "steps" not in record or "ground_truth" not in record:
            skipped += 1
            continue
        ckpt_b.append(process_trajectory(record, star_rationalize=True))

    if skipped > 0:
        print(f"  Skipped {skipped} malformed records in failed trajectories.")

    write_jsonl(ckpt_b, args.out_b)

    print("\nDone.")
    print(f"  Checkpoint A: {len(ckpt_a)} examples")
    print(f"  Checkpoint B: {len(ckpt_b)} examples  ({len(ckpt_b) - len(ckpt_a)} from failed trajectories)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data from trajectory JSONL files.")
    parser.add_argument("--success", default="success_traj.jsonl", help="Path to success trajectories")
    parser.add_argument("--failed",  default="failed_traj.jsonl",  help="Path to failed trajectories")
    parser.add_argument("--out_a",   default="sft_checkpoint_a.jsonl", help="Output path for Checkpoint A data")
    parser.add_argument("--out_b",   default="sft_checkpoint_b.jsonl", help="Output path for Checkpoint B data")
    args = parser.parse_args()
    main(args)
