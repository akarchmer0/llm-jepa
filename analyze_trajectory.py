"""Trajectory analysis for LLM-JEPA.

Tests whether hidden state trajectories in fine-tuned models have a parabolic
degeneracy: they stay approximately on a 1D line in embedding space, but are
traversed non-uniformly (accelerating/decelerating) rather than at constant speed.

Usage:
    python3 analyze_trajectory.py \
        --model_name ft-j-2e-5-0.02-0-82 \
        --original_model_name meta-llama/Llama-3.2-1B-Instruct \
        --input_file datasets/synth_test.jsonl \
        --max_examples 200
"""

import argparse
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, original_model_name=None):
    """Load model and tokenizer, using original_model_name for tokenizer if provided."""
    tokenizer_name = original_model_name if original_model_name else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading and span finding
# ---------------------------------------------------------------------------

def get_messages(model_name, messages):
    """Model-specific message reformatting (mirrors stp.py)."""
    if "google/gemma" in model_name:
        full_messages = copy.deepcopy(messages)[1:3]
        full_messages[0]["content"] = messages[0]["content"] + "\n\n" + full_messages[0]["content"]
        return full_messages
    return messages


def find_start_end(content, tokenizer, input_ids, attention_mask):
    """Find the start and end indices of content within input_ids.

    Returns (start - 1, end) matching stp.py convention so that
    user_start_end[0] + 1 = first token of the span.
    """
    tokens = tokenizer.encode(content, add_special_tokens=False)
    decoded_content = [tokenizer.decode(t) for t in tokens]
    decoded_input = [tokenizer.decode(t) for t in input_ids]

    # Scan from right (last occurrence) as in stp.py
    for i in range(len(input_ids) - len(tokens), -1, -1):
        if attention_mask[i] == 1 and decoded_input[i:i + len(tokens)] == decoded_content:
            assert i > 0, "Span starts at position 0, cannot subtract 1"
            return i - 1, i + len(tokens) - 1

    return None, None


def load_examples(input_file, model_name, tokenizer, max_length=2048, max_examples=None):
    """Load JSONL examples and return tokenized inputs with span boundaries."""
    examples = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break

    results = []
    for ex in examples:
        messages = ex["messages"]
        full_messages = get_messages(model_name, messages)

        formatted = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Find user span
        if "allenai/OLMo" in model_name:
            user_content = messages[1]["content"] + "\n"
        else:
            user_content = messages[1]["content"]
        user_start, user_end = find_start_end(user_content, tokenizer, input_ids, attention_mask)

        # Find assistant span
        if "apple/OpenELM" in model_name:
            try:
                asst_start, asst_end = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)
            except (AssertionError, TypeError):
                asst_start, asst_end = find_start_end("\n" + messages[2]["content"], tokenizer, input_ids, attention_mask)
        else:
            asst_start, asst_end = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)

        # Skip examples where span finding failed
        if any(x is None for x in [user_start, user_end, asst_start, asst_end]):
            continue

        results.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "user_start_end": (user_start, user_end),
            "asst_start_end": (asst_start, asst_end),
        })

    return results


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_trajectory(model, item, layer=-1):
    """Extract the hidden-state trajectory over the user+assistant span.

    Returns a float32 numpy array of shape (T, D).
    """
    device = next(model.parameters()).device
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    hs = outputs.hidden_states[layer][0]  # (seq_len, D)

    us, ue = item["user_start_end"]
    as_, ae = item["asst_start_end"]

    # stp.py convention: start_end[0]+1 = first token of span
    user_hs = hs[us + 1: ue + 1]
    asst_hs = hs[as_ + 1: ae + 1]

    if user_hs.shape[0] == 0 and asst_hs.shape[0] == 0:
        return None

    trajectory = torch.cat([user_hs, asst_hs], dim=0)  # (T, D)
    return trajectory.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def r2_score_high_d(H, X):
    """Compute R^2 for a multivariate least-squares fit H ~ X @ beta.

    H: (T, D)  response matrix
    X: (T, k)  design matrix (already includes intercept column)
    Returns scalar R^2 using Frobenius norm.
    """
    beta, _, _, _ = np.linalg.lstsq(X, H, rcond=None)   # (k, D)
    H_pred = X @ beta                                     # (T, D)
    H_mean = H.mean(axis=0, keepdims=True)

    ss_res = np.sum((H - H_pred) ** 2)
    ss_tot = np.sum((H - H_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(trajectory):
    """Compute all diagnostic metrics for a single trajectory.

    trajectory: numpy array of shape (T, D).
    Returns a dict of scalar metrics.
    """
    T, D = trajectory.shape
    if T < 3:
        return None  # too short for meaningful analysis

    t = np.linspace(0.0, 1.0, T)

    # ---- Linear and quadratic R^2 ----------------------------------------
    X_lin  = np.column_stack([np.ones(T), t])            # (T, 2)
    X_quad = np.column_stack([np.ones(T), t, t ** 2])    # (T, 3)
    lin_r2  = r2_score_high_d(trajectory, X_lin)
    quad_r2 = r2_score_high_d(trajectory, X_quad)
    r2_improvement = quad_r2 - lin_r2

    # ---- Velocity profile --------------------------------------------------
    velocities = trajectory[1:] - trajectory[:-1]          # (T-1, D)
    vel_norms = np.linalg.norm(velocities, axis=1)         # (T-1,)

    # Slope of velocity-norm vs time (negative = deceleration)
    t_vel = np.linspace(0.0, 1.0, T - 1)
    vel_slope = float(np.polyfit(t_vel, vel_norms, 1)[0])
    mean_vel_norm = float(vel_norms.mean())

    # ---- Acceleration ------------------------------------------------------
    accel = velocities[1:] - velocities[:-1]               # (T-2, D)
    accel_norms = np.linalg.norm(accel, axis=1)            # (T-2,)
    mean_accel_norm = float(accel_norms.mean()) if len(accel_norms) > 0 else 0.0

    # Alignment of acceleration vectors (mean pairwise cosine similarity)
    accel_alignment = 0.0
    if len(accel) >= 2:
        norms = accel_norms[:, None] + 1e-12               # avoid /0
        accel_unit = accel / norms                         # (T-2, D)
        cos_mat = accel_unit @ accel_unit.T                # (T-2, T-2)
        # Take upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(cos_mat, dtype=bool), k=1)
        if mask.sum() > 0:
            accel_alignment = float(cos_mat[mask].mean())

    # ---- PCA: variance explained by first PC -------------------------------
    H_centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(H_centered, full_matrices=False)
    total_var = float((S ** 2).sum())
    pc1_var = float(S[0] ** 2 / total_var) if total_var > 1e-12 else 0.0

    return {
        "T": T,
        "lin_r2": lin_r2,
        "quad_r2": quad_r2,
        "r2_improvement": r2_improvement,
        "vel_slope": vel_slope,
        "mean_vel_norm": mean_vel_norm,
        "mean_accel_norm": mean_accel_norm,
        "accel_alignment": accel_alignment,
        "pca_pc1_var": pc1_var,
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def summarize(all_metrics, label):
    keys = ["T", "lin_r2", "quad_r2", "r2_improvement",
            "vel_slope", "mean_vel_norm", "mean_accel_norm",
            "accel_alignment", "pca_pc1_var"]
    labels = {
        "T":               "Trajectory length (tokens)",
        "lin_r2":          "Linear R²",
        "quad_r2":         "Quadratic R²",
        "r2_improvement":  "R² improvement (quad - lin)",
        "vel_slope":       "Velocity slope (dspeed/dt)",
        "mean_vel_norm":   "Mean velocity norm",
        "mean_accel_norm": "Mean acceleration norm",
        "accel_alignment": "Accel alignment (cos)",
        "pca_pc1_var":     "PCA PC1 explained variance",
    }

    print(f"\n{'=' * 65}")
    print(f"  Trajectory Analysis: {label}")
    print(f"  Examples analyzed: {len(all_metrics)}")
    print(f"{'=' * 65}")
    print(f"{'Metric':<35} {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"{'-' * 65}")
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        print(f"{labels[k]:<35} {vals.mean():>8.4f}  {vals.std():>8.4f}  {np.median(vals):>8.4f}")
    print(f"{'=' * 65}")

    print("\nInterpretation guide:")
    print("  R² improvement >> 0  → quadratic fits better than linear (parabolic structure)")
    print("  Velocity slope  < 0  → systematic deceleration over the trajectory")
    print("  Accel norm     >> 0  → non-zero second derivative (non-linear traversal)")
    print("  Accel alignment ~ 1  → acceleration is consistent direction (parabola on a line)")
    print("  PCA PC1 var   > 0.95 → trajectory is approximately 1D")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze hidden-state trajectory geometry.")
    parser.add_argument("--model_name", required=True,
                        help="Path to fine-tuned model (or HuggingFace model ID).")
    parser.add_argument("--original_model_name", default=None,
                        help="Base model name for tokenizer (if different from --model_name).")
    parser.add_argument("--input_file", required=True,
                        help="Test JSONL file.")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which hidden state layer to analyze (default: -1 = last).")
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Maximum number of examples to analyze.")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max tokenized sequence length.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save per-example metrics as JSON.")
    args = parser.parse_args()

    label = args.model_name.split("/")[-1]
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.original_model_name)

    model_name_for_data = args.original_model_name or args.model_name
    print(f"Loading data: {args.input_file}")
    examples = load_examples(
        args.input_file, model_name_for_data, tokenizer,
        max_length=args.max_length, max_examples=args.max_examples
    )
    print(f"  Loaded {len(examples)} examples (after span-finding)")

    all_metrics = []
    skipped = 0
    for item in tqdm(examples, desc="Analyzing trajectories"):
        traj = extract_trajectory(model, item, layer=args.layer)
        if traj is None:
            skipped += 1
            continue
        metrics = compute_metrics(traj)
        if metrics is None:
            skipped += 1
            continue
        all_metrics.append(metrics)

    if skipped:
        print(f"  Skipped {skipped} examples (empty or too-short spans)")

    if not all_metrics:
        print("No valid trajectories found. Check --input_file and span-finding logic.")
        return

    summarize(all_metrics, label)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Per-example metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
