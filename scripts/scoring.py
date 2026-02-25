#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 2B — Composite Scoring Pipeline

Evaluates generated peptides using four orthogonal dimensions:

  1. Novelty       — 3-mer Jaccard dissimilarity vs. reference set
  2. Diversity     — mean pairwise 3-mer Jaccard distance within the batch
  3. OOD score     — Isolation Forest density in QSAR feature space
                     (high = closer to training distribution = better)
  4. AMP-QSAR score— rule-based score from physicochemical AMP features:
                     • charge_pH7 ∈ [+2, +9]     (cationic AMPs)
                     • gravy     ∈ [−1.0, +2.5]  (moderate hydrophobicity)
                     • h_moment_helix ≥ 0.35      (amphipathicity)
                     • boman_index ≥ 1.0          (membrane interaction)
                     • length    ∈ [8, 50]        (canonical AMP range)
                     • instability_index < 40     (stable peptides)

  Final composite score = weighted sum of the four components.
  Default weights: novelty=0.25, diversity=0.25, ood=0.25, amp_qsar=0.25

QSAR features can be:
  (a) pre-computed with qsar_extract.py (--qsar_csv flag)
  (b) computed inline on-the-fly (always available, slightly slower)

Usage
-----
    # Basic: generated CSV → scored CSV
    python scoring.py --input generated.csv --output scored.csv

    # With reference set for novelty
    python scoring.py --input gen.csv --output sc.csv \\
        --reference_csv training.csv

    # With pre-computed QSAR
    python scoring.py --input gen.csv --output sc.csv \\
        --qsar_csv qsar.csv --reference_csv train.csv

    # Custom weights
    python scoring.py --input gen.csv --output sc.csv \\
        --w_novelty 0.3 --w_diversity 0.2 --w_ood 0.2 --w_qsar 0.3

    # Filter output to top-k
    python scoring.py --input gen.csv --output sc.csv --top_k 100
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 3-mer helpers (consistent with Phase 1D novelty)
# ============================================================

def make_3mer_set(seq: str) -> Set[str]:
    s = str(seq).strip().upper()
    if len(s) < 3:
        return set()
    return {s[i: i + 3] for i in range(len(s) - 2)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    return 1.0 - jaccard_similarity(a, b)


# ============================================================
# 1. Novelty score
# ============================================================

def compute_novelty(
    gen_sets: List[Set[str]],
    ref_sets: List[Set[str]],
) -> np.ndarray:
    """
    For each generated peptide, novelty = 1 - max_Jaccard(gen, ref).
    Returns array of shape [N] with values in [0, 1].
    1.0 = completely novel; 0.0 = identical to a reference.
    """
    if not ref_sets:
        return np.ones(len(gen_sets), dtype=np.float32)

    scores = np.zeros(len(gen_sets), dtype=np.float32)
    for i, gs in enumerate(gen_sets):
        if not gs:
            scores[i] = 1.0
            continue
        max_sim = 0.0
        for rs in ref_sets:
            if not rs:
                continue
            sim = jaccard_similarity(gs, rs)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= 0.999:
                break
        scores[i] = 1.0 - max_sim
    return scores


# ============================================================
# 2. Diversity score (intra-batch)
# ============================================================

def compute_diversity(gen_sets: List[Set[str]]) -> np.ndarray:
    """
    For each peptide, diversity = mean pairwise Jaccard distance to all others.
    Returns array of shape [N] with values in [0, 1].
    1.0 = maximally diverse; 0.0 = identical to all others.
    """
    N = len(gen_sets)
    if N <= 1:
        return np.ones(N, dtype=np.float32)

    # Build similarity matrix (upper triangle)
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            s = jaccard_similarity(gen_sets[i], gen_sets[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    # Mean pairwise DISTANCE (not similarity)
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    div = dist_matrix.sum(axis=1) / (N - 1)
    return div.astype(np.float32)


# ============================================================
# 3. OOD score via Isolation Forest
# ============================================================

def compute_ood_score(
    gen_features: np.ndarray,
    ref_features: Optional[np.ndarray] = None,
    n_estimators: int = 200,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit Isolation Forest on ref_features (or gen_features if no ref).
    Return anomaly score mapped to [0, 1] where:
      1.0 = inlier (close to training distribution, preferred)
      0.0 = outlier / OOD

    Uses sklearn's decision_function (higher = more normal).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    N = len(gen_features)
    if N == 0:
        return np.array([], dtype=np.float32)

    # Fill NaN with column medians
    def fill_nan(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy().astype(np.float32)
        col_medians = np.nanmedian(arr, axis=0)
        for j in range(arr.shape[1]):
            nans = np.isnan(arr[:, j])
            arr[nans, j] = col_medians[j]
        return arr

    gen_clean = fill_nan(gen_features)

    if ref_features is not None and len(ref_features) > 0:
        ref_clean = fill_nan(ref_features)
        scaler = RobustScaler().fit(ref_clean)
        X_train = scaler.transform(ref_clean)
        X_test  = scaler.transform(gen_clean)
    else:
        scaler = RobustScaler().fit(gen_clean)
        X_train = scaler.transform(gen_clean)
        X_test  = X_train

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train)

    # decision_function: positive = more normal, negative = anomalous
    raw = clf.decision_function(X_test)
    # Map to [0, 1]: normalize by min/max of training scores
    train_raw = clf.decision_function(X_train)
    lo, hi = train_raw.min(), train_raw.max()
    if hi - lo < 1e-8:
        return np.ones(N, dtype=np.float32) * 0.5
    scores = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    return scores.astype(np.float32)


# ============================================================
# 4. AMP-QSAR score (rule-based)
# ============================================================

def compute_amp_qsar_score(features_df: pd.DataFrame) -> np.ndarray:
    """
    Continuous AMP-QSAR score based on empirical AMP property ranges.

    Each criterion yields a soft sigmoid score in [0, 1]:
    - charge_pH7 ∈ [+2, +9]        (optimal: ~+4)
    - gravy      ∈ [-1.0, +2.5]    (moderate hydrophobicity)
    - h_moment_helix ≥ 0.35        (amphipathicity threshold)
    - boman_index ≥ 1.0            (membrane affinity)
    - length ∈ [8, 50]             (size filter; hard boundary)
    - instability_index < 40       (stability; hard boundary)

    Returns scores in [0, 1] (mean of sub-scores).
    """

    def sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x - center) / width))

    N = len(features_df)
    scores = np.zeros((N, 6), dtype=np.float32)

    # 1. Charge: sigmoid centred at 4, peak in [2,9]
    if "charge_pH7" in features_df.columns:
        q = features_df["charge_pH7"].fillna(0).values
        # score high for q in [2, 9]: bell around 4-6
        scores[:, 0] = (
            sigmoid(q, center=2.0, width=0.8) *
            (1.0 - sigmoid(q, center=9.0, width=0.8))
        )

    # 2. Hydrophobicity GRAVY: bell centred at ~0.5 (slight hydrophobic)
    if "gravy" in features_df.columns:
        g = features_df["gravy"].fillna(0).values
        scores[:, 1] = (
            sigmoid(g, center=-1.0, width=0.3) *
            (1.0 - sigmoid(g, center=2.5, width=0.3))
        )

    # 3. Hydrophobic moment: sigmoid at 0.35
    if "h_moment_helix" in features_df.columns:
        hm = features_df["h_moment_helix"].fillna(0).values
        scores[:, 2] = sigmoid(hm, center=0.35, width=0.05)

    # 4. Boman index: sigmoid at 1.0
    if "boman_index" in features_df.columns:
        bi = features_df["boman_index"].fillna(0).values
        scores[:, 3] = sigmoid(bi, center=1.0, width=0.5)

    # 5. Length: hard filter [8, 50], soft edges
    if "length" in features_df.columns:
        ln = features_df["length"].fillna(0).values
        scores[:, 4] = (
            sigmoid(ln, center=8.0, width=1.0) *
            (1.0 - sigmoid(ln, center=50.0, width=1.0))
        )

    # 6. Instability: penalise II > 40
    if "instability_index" in features_df.columns:
        ii = features_df["instability_index"].fillna(50).values
        scores[:, 5] = 1.0 - sigmoid(ii, center=40.0, width=3.0)

    # Mean of available sub-scores (only columns that had data)
    active = (scores.sum(axis=0) > 0)
    if active.sum() == 0:
        return np.full(N, 0.5, dtype=np.float32)
    return scores[:, active].mean(axis=1)


# ============================================================
# QSAR feature columns used for OOD
# ============================================================

QSAR_FEATURE_COLS = [
    "length", "mw", "charge_pH7", "pI", "gravy", "mean_eisenberg",
    "h_moment_helix", "h_moment_beta", "boman_index", "aliphatic_index",
    "instability_index", "frac_positive", "frac_negative", "frac_charged",
    "frac_polar", "frac_apolar", "frac_aromatic", "frac_cys", "frac_pro",
    "frac_gly", "net_charge_ratio", "charge_hydro_ratio",
]


# ============================================================
# Inline QSAR computation (imports qsar_extract if available)
# ============================================================

def _get_qsar_inline(sequences: List[str], ph: float = 7.0) -> pd.DataFrame:
    """Compute QSAR features inline without external script."""
    try:
        # Try importing from the same package / scripts dir
        import sys
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from qsar_extract import compute_qsar_batch
        return compute_qsar_batch(sequences, ph=ph, verbose=True)
    except ImportError:
        raise ImportError(
            "qsar_extract.py not found. Either run qsar_extract.py first "
            "and pass --qsar_csv, or ensure qsar_extract.py is in the same "
            "directory as scoring.py."
        )


# ============================================================
# Composite score
# ============================================================

def composite_score(
    novelty: np.ndarray,
    diversity: np.ndarray,
    ood: np.ndarray,
    amp_qsar: np.ndarray,
    w_novelty: float = 0.25,
    w_diversity: float = 0.25,
    w_ood: float = 0.25,
    w_qsar: float = 0.25,
) -> np.ndarray:
    total_w = w_novelty + w_diversity + w_ood + w_qsar
    if total_w <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return (
        w_novelty  * novelty +
        w_diversity * diversity +
        w_ood      * ood +
        w_qsar     * amp_qsar
    ) / total_w


# ============================================================
# Main pipeline
# ============================================================

def run_scoring(
    gen_df: pd.DataFrame,
    sequence_col: str = "sequence",
    ref_df: Optional[pd.DataFrame] = None,
    qsar_df: Optional[pd.DataFrame] = None,
    ref_qsar_df: Optional[pd.DataFrame] = None,
    ph: float = 7.0,
    w_novelty: float = 0.25,
    w_diversity: float = 0.25,
    w_ood: float = 0.25,
    w_qsar: float = 0.25,
    ood_n_estimators: int = 200,
    ood_contamination: float = 0.1,
) -> pd.DataFrame:
    """
    Full scoring pipeline. Returns gen_df with score columns added.
    """
    sequences = gen_df[sequence_col].astype(str).str.strip().str.upper().tolist()
    N = len(sequences)
    print(f"[INFO] Scoring {N} sequences")

    # ---- QSAR features ----
    if qsar_df is None:
        print("[INFO] Computing QSAR inline...")
        qsar_df = _get_qsar_inline(sequences, ph=ph)
    else:
        qsar_df = qsar_df.reset_index(drop=True)
        print(f"[INFO] Using pre-computed QSAR ({len(qsar_df)} rows)")

    # Align by position
    if len(qsar_df) != N:
        raise ValueError(
            f"QSAR rows ({len(qsar_df)}) != sequence count ({N}). "
            "Ensure --qsar_csv matches --input row-for-row."
        )

    # ---- 3-mer sets ----
    print("[INFO] Building 3-mer sets...")
    gen_sets = [make_3mer_set(s) for s in sequences]

    ref_sets: List[Set[str]] = []
    if ref_df is not None and sequence_col in ref_df.columns:
        ref_seqs = ref_df[sequence_col].astype(str).str.strip().str.upper().tolist()
        ref_sets = [make_3mer_set(s) for s in ref_seqs]
        print(f"[INFO] Reference set: {len(ref_sets)} sequences")

    # ---- 1. Novelty ----
    print("[INFO] Computing novelty...")
    nov = compute_novelty(gen_sets, ref_sets)

    # ---- 2. Diversity ----
    print("[INFO] Computing diversity (pairwise)...")
    div = compute_diversity(gen_sets)

    # ---- 3. OOD ----
    print("[INFO] Computing OOD score (Isolation Forest)...")
    avail_cols = [c for c in QSAR_FEATURE_COLS if c in qsar_df.columns]
    gen_feat = qsar_df[avail_cols].values.astype(np.float32)

    ref_feat: Optional[np.ndarray] = None
    if ref_qsar_df is not None:
        ref_avail = [c for c in avail_cols if c in ref_qsar_df.columns]
        ref_feat = ref_qsar_df[ref_avail].values.astype(np.float32)
    elif ref_df is not None:
        print("[INFO] Computing QSAR for reference set (OOD training)...")
        ref_seqs_list = ref_df[sequence_col].astype(str).str.strip().str.upper().tolist()
        ref_qsar_inline = _get_qsar_inline(ref_seqs_list, ph=ph)
        ref_avail = [c for c in avail_cols if c in ref_qsar_inline.columns]
        ref_feat = ref_qsar_inline[ref_avail].values.astype(np.float32)

    ood = compute_ood_score(
        gen_feat,
        ref_features=ref_feat,
        n_estimators=ood_n_estimators,
        contamination=ood_contamination,
    )

    # ---- 4. AMP-QSAR ----
    print("[INFO] Computing AMP-QSAR score...")
    amp_q = compute_amp_qsar_score(qsar_df)

    # ---- Composite ----
    print("[INFO] Computing composite score...")
    comp = composite_score(nov, div, ood, amp_q,
                           w_novelty=w_novelty,
                           w_diversity=w_diversity,
                           w_ood=w_ood,
                           w_qsar=w_qsar)

    # ---- Assemble output ----
    out = gen_df.copy().reset_index(drop=True)

    # Append QSAR cols not already present
    for col in qsar_df.columns:
        if col not in out.columns:
            out[col] = qsar_df[col].values

    out["score_novelty"]   = nov
    out["score_diversity"] = div
    out["score_ood"]       = ood
    out["score_amp_qsar"]  = amp_q
    out["score_composite"] = comp
    out["rank"]            = out["score_composite"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank").reset_index(drop=True)

    return out


# ============================================================
# Summary reporting
# ============================================================

def print_summary(scored: pd.DataFrame, top_k: int = 10):
    print("\n" + "=" * 60)
    print("SCORING SUMMARY")
    print("=" * 60)

    score_cols = ["score_novelty", "score_diversity", "score_ood", "score_amp_qsar", "score_composite"]
    for col in score_cols:
        if col in scored.columns:
            v = scored[col].dropna()
            print(f"  {col:22s}: mean={v.mean():.3f}  std={v.std():.3f}  "
                  f"[{v.min():.3f}, {v.max():.3f}]")

    print(f"\nTop-{top_k} candidates (by composite score):")
    show_cols = ["sequence"]
    for c in ["bacteria", "length", "charge_pH7", "gravy", "h_moment_helix",
              "score_composite", "score_novelty", "score_amp_qsar"]:
        if c in scored.columns:
            show_cols.append(c)

    top = scored.head(top_k)[show_cols]
    # Format floats nicely
    float_cols = [c for c in top.columns if top[c].dtype == np.float64 or top[c].dtype == np.float32]
    fmt = {c: "{:.3f}".format for c in float_cols}
    print(top.to_string(index=False, formatters=fmt))
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 2B — Composite scoring for generated peptides"
    )
    # I/O
    p.add_argument("--input", required=True,
                   help="Generated peptides CSV (output of any Phase 1D script)")
    p.add_argument("--output", required=True,
                   help="Output CSV with scoring columns appended")
    p.add_argument("--sequence_col", default="sequence",
                   help="Name of sequence column (default: sequence)")

    # Reference
    p.add_argument("--reference_csv", default=None,
                   help="Reference/training CSV for novelty + OOD (optional)")
    p.add_argument("--reference_max", type=int, default=0,
                   help="Max reference sequences to use (0 = all)")

    # Pre-computed QSAR
    p.add_argument("--qsar_csv", default=None,
                   help="Pre-computed QSAR features from qsar_extract.py (optional)")
    p.add_argument("--ref_qsar_csv", default=None,
                   help="Pre-computed QSAR for reference set (optional)")

    # Score weights
    p.add_argument("--w_novelty",   type=float, default=0.25)
    p.add_argument("--w_diversity", type=float, default=0.25)
    p.add_argument("--w_ood",       type=float, default=0.25)
    p.add_argument("--w_qsar",      type=float, default=0.25)

    # OOD hyperparams
    p.add_argument("--ood_n_estimators", type=int,   default=200)
    p.add_argument("--ood_contamination", type=float, default=0.1)

    # Output options
    p.add_argument("--top_k",  type=int,   default=0,
                   help="Keep only top-k sequences in output (0 = keep all)")
    p.add_argument("--ph",     type=float, default=7.0,
                   help="pH for inline QSAR computation (default: 7.0)")

    args = p.parse_args()

    # Load generated peptides
    print(f"[INFO] Loading generated peptides: {args.input}")
    gen_df = pd.read_csv(args.input)
    gen_df.columns = [c.strip() for c in gen_df.columns]

    # Normalise sequence col
    if args.sequence_col not in gen_df.columns:
        lc = {c.lower(): c for c in gen_df.columns}
        if args.sequence_col.lower() in lc:
            args.sequence_col = lc[args.sequence_col.lower()]
        else:
            raise ValueError(
                f"Column '{args.sequence_col}' not found. "
                f"Available: {list(gen_df.columns)}"
            )

    print(f"[INFO] Sequences: {len(gen_df)}")

    # Load reference
    ref_df = None
    if args.reference_csv and os.path.exists(args.reference_csv):
        ref_df = pd.read_csv(args.reference_csv)
        ref_df.columns = [c.strip() for c in ref_df.columns]
        if args.sequence_col not in ref_df.columns:
            lc = {c.lower(): c for c in ref_df.columns}
            sc = args.sequence_col.lower()
            if sc in lc:
                ref_df = ref_df.rename(columns={lc[sc]: args.sequence_col})
        if args.reference_max and args.reference_max > 0:
            ref_df = ref_df.head(int(args.reference_max))
        print(f"[INFO] Reference: {len(ref_df)} sequences")

    # Load pre-computed QSAR
    qsar_df = None
    if args.qsar_csv and os.path.exists(args.qsar_csv):
        qsar_df = pd.read_csv(args.qsar_csv)
        print(f"[INFO] QSAR CSV: {len(qsar_df)} rows, {len(qsar_df.columns)} cols")
        # Keep only QSAR feature columns
        keep = [c for c in QSAR_FEATURE_COLS if c in qsar_df.columns]
        qsar_df = qsar_df[keep].reset_index(drop=True)

    ref_qsar_df = None
    if args.ref_qsar_csv and os.path.exists(args.ref_qsar_csv):
        ref_qsar_df = pd.read_csv(args.ref_qsar_csv)
        keep = [c for c in QSAR_FEATURE_COLS if c in ref_qsar_df.columns]
        ref_qsar_df = ref_qsar_df[keep].reset_index(drop=True)
        print(f"[INFO] Reference QSAR CSV: {len(ref_qsar_df)} rows")

    # Run scoring
    scored = run_scoring(
        gen_df=gen_df,
        sequence_col=args.sequence_col,
        ref_df=ref_df,
        qsar_df=qsar_df,
        ref_qsar_df=ref_qsar_df,
        ph=args.ph,
        w_novelty=args.w_novelty,
        w_diversity=args.w_diversity,
        w_ood=args.w_ood,
        w_qsar=args.w_qsar,
        ood_n_estimators=args.ood_n_estimators,
        ood_contamination=args.ood_contamination,
    )

    # Optional top-k filter
    if args.top_k and args.top_k > 0:
        before = len(scored)
        scored = scored.head(int(args.top_k))
        print(f"[INFO] Filtered to top-{args.top_k} (from {before})")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"\n[OK] Saved: {args.output} | rows={len(scored)}")

    print_summary(scored, top_k=min(10, len(scored)))


if __name__ == "__main__":
    main()
