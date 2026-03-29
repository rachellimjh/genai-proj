"""
verify_trajectories.py — Check trajectories for correctness and export results
===============================================================================
Reads trajectories.jsonl, checks each answer against ground truth,
and splits into success/failed files with statistics.

Usage:
  python verify_trajectories.py
  python verify_trajectories.py --path /custom/path/trajectories.jsonl
  python verify_trajectories.py --show-failures 5
"""

import json
import re
import os
import argparse
from collections import Counter


# =============================================================================
# Answer matching
# =============================================================================

def normalize(s):
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def check_answer(prediction, ground_truth):
    """
    Check if the prediction matches the ground truth.
    
    Handles:
      - ground_truth as a string or list of acceptable answers
      - Normalized exact match (primary)
      - Containment check as fallback (ground truth found within prediction)
    
    Returns: (is_correct: bool, match_type: str or None)
    """
    if ground_truth is None or ground_truth == "":
        return None, None  # Can't evaluate
    
    pred_norm = normalize(prediction)
    
    if not pred_norm:
        return False, 'no_answer_text'
    
    # Handle list of acceptable answers
    gt_list = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    
    for gt in gt_list:
        gt_norm = normalize(gt)
        if not gt_norm:
            continue
        
        # Exact match (after normalization)
        if pred_norm == gt_norm:
            return True, 'exact'
        
        # Containment: ground truth appears as a substring of prediction
        # Use word boundaries to avoid "1" matching "1920"
        pattern = r'\b' + re.escape(gt_norm) + r'\b'
        if re.search(pattern, pred_norm):
            return True, 'contains'
    
    return False, None


def f1_score(prediction, ground_truth):
    """Token-level F1 between prediction and best-matching ground truth."""
    if ground_truth is None:
        return 0.0
    
    gt_list = ground_truth if isinstance(ground_truth, (list, tuple)) else [ground_truth]
    
    best_f1 = 0.0
    pred_tokens = normalize(prediction).split()
    
    for gt in gt_list:
        gt_tokens = normalize(gt).split()
        if not pred_tokens or not gt_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    
    return best_f1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify trajectories")
    parser.add_argument('--path', type=str,
                        default=os.path.expanduser('~/Search-R1/trajectories.jsonl'))
    parser.add_argument('--show-failures', type=int, default=3,
                        help='Number of failure examples to print')
    args = parser.parse_args()

    input_path = args.path
    base_dir = os.path.dirname(input_path)
    success_path = os.path.join(base_dir, 'success_traj.jsonl')
    failed_path = os.path.join(base_dir, 'failed_traj.jsonl')

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # ── Load trajectories ────────────────────────────────────────────────
    trajectories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                trajectories.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}")

    print(f"Loaded {len(trajectories)} trajectories from {input_path}")
    print("=" * 70)

    if not trajectories:
        return

    # ── Evaluate each trajectory ─────────────────────────────────────────
    successful_trajs = []
    failed_trajs = []
    unevaluable = 0
    
    match_types = Counter()       # exact, contains
    failure_modes = Counter()     # no_answer, no_search, wrong_answer, etc.
    f1_scores = []
    
    all_results = []  # for detailed analysis

    for traj in trajectories:
        ground_truth = traj.get('ground_truth')
        steps = traj.get('steps', [])
        
        # Extract answer
        answer_text = ""
        for step in steps:
            if step.get('action') == 'answer':
                answer_text = step.get('content', '')
                break
        
        # Count actions
        num_thinks = sum(1 for s in steps if s.get('action') == 'think')
        num_searches = sum(1 for s in steps if s.get('action') == 'search')
        has_search_results = any(
            s.get('search_results') and str(s['search_results']).strip()
            for s in steps if s.get('action') == 'search'
        )
        
        # Check correctness
        is_correct, match_type = check_answer(answer_text, ground_truth)
        f1 = f1_score(answer_text, ground_truth)
        f1_scores.append(f1)
        
        # Classify
        if is_correct is None:
            unevaluable += 1
            failed_trajs.append(traj)  # Can't verify = treat as failed
            failure_modes['no_ground_truth'] += 1
        elif is_correct:
            successful_trajs.append(traj)
            match_types[match_type] += 1
        else:
            failed_trajs.append(traj)
            # Classify failure mode
            if not answer_text.strip():
                failure_modes['no_answer'] += 1
            elif num_searches == 0:
                failure_modes['no_search'] += 1
            elif not has_search_results:
                failure_modes['empty_search_results'] += 1
            else:
                failure_modes['wrong_answer'] += 1
        
        all_results.append({
            'question': traj.get('question', '')[:100],
            'ground_truth': ground_truth,
            'answer': answer_text[:100],
            'is_correct': is_correct,
            'match_type': match_type,
            'f1': f1,
            'num_thinks': num_thinks,
            'num_searches': num_searches,
            'has_search_results': has_search_results,
        })

    # ── Print statistics ─────────────────────────────────────────────────
    total = len(trajectories)
    evaluable = total - unevaluable

    print(f"\nRESULTS")
    print(f"{'─' * 70}")
    print(f"  Total trajectories:    {total}")
    print(f"  Evaluable:             {evaluable}")
    print(f"  Unevaluable (no GT):   {unevaluable}")
    print(f"  Correct:               {len(successful_trajs)}/{total} ({100*len(successful_trajs)/total:.1f}%)")
    if evaluable > 0:
        print(f"  Accuracy (evaluable):  {len(successful_trajs)}/{evaluable} ({100*len(successful_trajs)/evaluable:.1f}%)")
    if f1_scores:
        print(f"  Average F1:            {sum(f1_scores)/len(f1_scores):.3f}")

    # Match type breakdown
    if match_types:
        print(f"\n  Match types (correct episodes):")
        for mtype, count in match_types.most_common():
            print(f"    {mtype}: {count}")

    # Failure mode breakdown
    if failure_modes:
        print(f"\n  Failure modes:")
        for mode, count in failure_modes.most_common():
            print(f"    {mode}: {count}")

    # Search usage stats
    print(f"\n  Search statistics:")
    trajs_with_search = sum(1 for r in all_results if r['num_searches'] > 0)
    print(f"    Trajectories with search: {trajs_with_search}/{total}")
    print(f"    Avg searches/trajectory:  {sum(r['num_searches'] for r in all_results)/total:.2f}")
    
    search_dist = Counter(r['num_searches'] for r in all_results)
    for count in sorted(search_dist.keys()):
        n = search_dist[count]
        print(f"    {count} searches: {n} trajectories")

    # Correct vs incorrect comparison
    correct_results = [r for r in all_results if r['is_correct'] is True]
    incorrect_results = [r for r in all_results if r['is_correct'] is False]
    
    if correct_results and incorrect_results:
        print(f"\n  Correct vs Incorrect:")
        print(f"    {'Metric':<30}{'Correct':<15}{'Incorrect':<15}")
        print(f"    {'─'*30}{'─'*15}{'─'*15}")
        
        avg_s_c = sum(r['num_searches'] for r in correct_results) / len(correct_results)
        avg_s_i = sum(r['num_searches'] for r in incorrect_results) / len(incorrect_results)
        print(f"    {'Avg searches':<30}{avg_s_c:<15.2f}{avg_s_i:<15.2f}")
        
        avg_f1_c = sum(r['f1'] for r in correct_results) / len(correct_results)
        avg_f1_i = sum(r['f1'] for r in incorrect_results) / len(incorrect_results)
        print(f"    {'Avg F1':<30}{avg_f1_c:<15.3f}{avg_f1_i:<15.3f}")
        
        pct_sr_c = 100 * sum(1 for r in correct_results if r['has_search_results']) / len(correct_results)
        pct_sr_i = 100 * sum(1 for r in incorrect_results if r['has_search_results']) / len(incorrect_results)
        print(f"    {'% with search results':<30}{pct_sr_c:<14.1f}%{pct_sr_i:<14.1f}%")

    # ── Export ───────────────────────────────────────────────────────────
    with open(success_path, 'w', encoding='utf-8') as f:
        for traj in successful_trajs:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    with open(failed_path, 'w', encoding='utf-8') as f:
        for traj in failed_trajs:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(successful_trajs)} successful to: {success_path}")
    print(f"Saved {len(failed_trajs)} failed to: {failed_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()