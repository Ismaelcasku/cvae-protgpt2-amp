#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 2A — QSAR Feature Extraction for Peptides

Computes physicochemical and sequence-based descriptors used as
input to the Structural+QSAR Classifier and Scoring pipeline.
No dependencies beyond numpy and pandas.

Features computed
-----------------
- length              : sequence length
- mw                  : molecular weight (Da), linear peptide
- charge_pH7          : net charge at pH 7.0 (Henderson-Hasselbalch)
- pI                  : isoelectric point (binary search, ±0.01 pH units)
- gravy               : GRAVY index (Kyte-Doolittle, 1982)
- mean_eisenberg       : mean Eisenberg consensus hydrophobicity
- h_moment_helix       : hydrophobic moment µH (α-helix, δ=100°), Eisenberg 1982
- h_moment_beta        : hydrophobic moment µH (β-sheet, δ=160°)
- boman_index          : sum of transfer energies (Boman 2003)
- aliphatic_index      : aliphatic index (Ikai 1980)
- instability_index    : instability index (Guruprasad et al. 1990)
- frac_positive        : fraction of K+R+H residues
- frac_negative        : fraction of D+E residues
- frac_charged         : frac_positive + frac_negative
- frac_polar           : fraction of N+Q+S+T+Y residues
- frac_apolar          : fraction of A+G+V+L+I+P+F+M+W+C residues
- frac_aromatic        : fraction of F+W+Y residues
- frac_cys             : fraction of C residues
- frac_pro             : fraction of P residues
- frac_gly             : fraction of G residues (flexibility)
- net_charge_ratio     : charge_pH7 / length (charge density)
- charge_hydro_ratio   : charge_pH7 / (|gravy| + 1e-3)  (Wieprecht 1999)

Usage
-----
    python qsar_extract.py --input generated.csv --output qsar.csv
    python qsar_extract.py --input gen.csv --output qsar.csv --ph 7.4
    python qsar_extract.py --input gen.csv --output qsar.csv --sequence_col seq
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ============================================================
# Amino acid property scales
# ============================================================

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte & Doolittle hydrophobicity (J. Mol. Biol. 157:105, 1982)
KD_HYDRO: Dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Eisenberg consensus hydrophobicity (PNAS 81:140, 1984)
# Used for hydrophobic moment calculation
EISENBERG_HYDRO: Dict[str, float] = {
    "A":  0.620, "R": -2.530, "N": -0.780, "D": -0.900, "C":  0.290,
    "Q": -0.850, "E": -0.740, "G":  0.480, "H": -0.400, "I":  1.380,
    "L":  1.060, "K": -1.500, "M":  0.640, "F":  1.190, "P":  0.120,
    "S": -0.180, "T": -0.050, "W":  0.810, "Y":  0.260, "V":  1.080,
}

# Boman index — transfer energies (Boman, J Pept Sci 9:700, 2003)
BOMAN: Dict[str, float] = {
    "A":  0.17, "R":  0.81, "N":  0.42, "D":  1.23, "C": -0.24,
    "Q":  0.58, "E":  2.02, "G":  0.01, "H":  0.96, "I": -1.56,
    "L": -1.18, "K":  0.99, "M": -0.01, "F": -1.71, "P":  0.45,
    "S":  0.13, "T":  0.14, "W": -2.09, "Y": -0.71, "V": -1.27,
}

# Average residue molecular weights (Da) — free amino acids minus water
AA_MW: Dict[str, float] = {
    "A":  89.094, "R": 174.203, "N": 132.119, "D": 133.104, "C": 121.159,
    "Q": 146.146, "E": 147.130, "G":  75.067, "H": 155.156, "I": 131.175,
    "L": 131.175, "K": 146.189, "M": 149.208, "F": 165.192, "P": 115.132,
    "S": 105.093, "T": 119.119, "W": 204.228, "Y": 181.191, "V": 117.148,
}
MW_WATER = 18.015  # added once per linear peptide

# pKa values for charge computation (Bjellqvist et al. / Lehninger)
PKA_NTERM: float = 8.0
PKA_CTERM: float = 3.1
PKA_SIDE: Dict[str, float] = {
    "D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07,
    "H": 6.00, "K": 10.53, "R": 12.48,
}
# -1 = acidic (loses proton above pKa) ; +1 = basic (gains proton below pKa)
PKA_SIGN: Dict[str, int] = {
    "D": -1, "E": -1, "C": -1, "Y": -1,
    "H": +1, "K": +1, "R": +1,
}

# Dipeptide Instability Weight Values — Guruprasad et al., Protein Eng. 4:155, 1990
# Default value for unlisted pairs = 1.0
DIWV: Dict[str, float] = {
    "WC": -1.0,  "WD": -1.0,  "WE": -1.0,  "WG": -1.0,  "WH": -1.0,
    "WK": -1.0,  "WL": 13.34, "WM": 24.68, "WN": 13.34, "WT": -14.03,
    "WV": -7.49,
    "CK":  1.0,  "CM": 33.60, "CP": 20.26, "CQ": -6.54, "CT": 33.60,
    "CV": -6.54, "CW": 24.68,
    "AC": 44.94, "AD": -7.49, "AG":  1.0,  "AH": -7.49, "AK": -7.49,
    "AP": 20.26,
    "EC": 44.94, "EE": -6.54, "EG": -6.54, "EH": -6.54, "EM": -6.54,
    "EQ": -6.54, "ES": -6.54, "ET": -6.54, "EW": -14.03, "EP": 20.26,
    "DS": 20.26,
    "GG": 13.34, "GI": -7.49, "GK": -7.49, "GN": -7.49, "GT": -7.49,
    "GW": 13.34, "GY": -7.49, "GA": -7.49, "GE": -6.54,
    "FD": 13.34, "FP": 20.26, "FY": 33.60,
    "IE": 44.94, "IH": -7.49, "IK": -7.49, "IL": 20.26, "IP": -1.88,
    "IV": -7.49,
    "HG": -9.37, "HI": 44.94, "HN": 24.68, "HP": -1.88, "HW": -1.88,
    "HY": 44.94,
    "KC": 1.0,  "KG": -7.49, "KI": -7.49, "KK":  1.0,  "KL": -7.49,
    "KM": 33.60, "KP": -6.54, "KQ": 24.68, "KR": 33.60, "KV": -7.49,
    "LC": 1.0,  "LL": 1.0,  "LP": 20.26, "LQ": 33.60, "LR": 20.26,
    "LK": -7.49, "LW": 24.68,
    "MH": 58.28, "MK": 33.60, "MM": -1.88, "MP": 44.94, "MQ": -6.54,
    "MR": -6.54, "MS": 44.94, "MT": -1.88, "MY": 24.68, "MA": 13.34,
    "NG": -14.03, "NF": -14.03, "NI": 44.94, "NP": -1.88, "NQ": -6.54,
    "NT": -7.49, "NW": -9.37,
    "PE": 18.38, "PF": 20.26, "PG": -6.54, "PK": -6.54, "PM": -6.54,
    "PP": 20.26, "PQ": 20.26, "PR": -6.54, "PS": 20.26, "PV": 20.26,
    "PW": -1.88, "PC": -6.54, "PD": -6.54, "PA": 20.26,
    "QC": -6.54, "QD": 20.26, "QE": -6.54, "QF": -6.54, "QQ": 20.26,
    "QS": 44.94, "QV": -6.54, "QY": -6.54, "QP": 20.26,
    "RG": -7.49, "RH": 20.26, "RP": 20.26, "RQ": 20.26, "RR": 58.28,
    "RS": 44.94, "RW": 58.28, "RY": -6.54,
    "SC": 33.60, "SE": 20.26, "SP": 44.94, "SQ": 20.26, "SR": 20.26,
    "SS": 20.26,
    "TF": 13.34, "TG": -7.49, "TN": -14.03, "TQ": -6.54, "TW": -14.03,
    "VD": -7.49, "VG": -7.49, "VN": -7.49, "VP": 20.26, "VT": -7.49,
    "VY": -6.54,
    "YE": -6.54, "YG": -7.49, "YH": 13.34, "YM": 44.94, "YP": 13.34,
    "YR": -15.91, "YT": -7.49, "YW": -9.37, "YY": 13.34,
}


# ============================================================
# Charge / pI helpers
# ============================================================

def _charge_at_ph(seq: str, ph: float) -> float:
    """Net charge of a linear peptide at the given pH."""
    # N-terminus (basic, pKa ~8.0)
    charge = 1.0 / (1.0 + 10.0 ** (ph - PKA_NTERM))
    # C-terminus (acidic, pKa ~3.1)
    charge -= 1.0 / (1.0 + 10.0 ** (PKA_CTERM - ph))
    for aa in seq:
        if aa in PKA_SIDE:
            pka = PKA_SIDE[aa]
            sign = PKA_SIGN[aa]
            if sign == +1:   # basic sidechain
                charge += 1.0 / (1.0 + 10.0 ** (ph - pka))
            else:            # acidic sidechain
                charge -= 1.0 / (1.0 + 10.0 ** (pka - ph))
    return charge


def charge_at_ph7(seq: str, ph: float = 7.0) -> float:
    return _charge_at_ph(seq, ph)


def isoelectric_point(seq: str, tol: float = 0.01) -> float:
    """Binary search for pI (net charge ≈ 0)."""
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        q = _charge_at_ph(seq, mid)
        if abs(q) < tol:
            return mid
        if q > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return (lo + hi) / 2.0


# ============================================================
# Hydrophobicity / GRAVY
# ============================================================

def gravy(seq: str) -> float:
    """Grand Average of Hydropathicity (Kyte-Doolittle)."""
    vals = [KD_HYDRO[aa] for aa in seq if aa in KD_HYDRO]
    return float(np.mean(vals)) if vals else 0.0


def mean_eisenberg(seq: str) -> float:
    vals = [EISENBERG_HYDRO[aa] for aa in seq if aa in EISENBERG_HYDRO]
    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# Hydrophobic moment (Eisenberg et al., 1982)
# ============================================================

def hydrophobic_moment(seq: str, delta_deg: float = 100.0, window: int = 11) -> float:
    """
    Maximum hydrophobic moment over all windows of given size.
    δ = 100° → alpha-helix ; δ = 160° → beta-sheet.
    Returns the maximum µH across all windows.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    delta = math.radians(delta_deg)
    hs = [EISENBERG_HYDRO.get(aa, 0.0) for aa in seq]
    w = min(window, n)
    max_mu = 0.0
    for start in range(n - w + 1):
        block = hs[start: start + w]
        angles = [i * delta for i in range(w)]
        sin_sum = sum(h * math.sin(a) for h, a in zip(block, angles))
        cos_sum = sum(h * math.cos(a) for h, a in zip(block, angles))
        mu = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / w
        if mu > max_mu:
            max_mu = mu
    return float(max_mu)


# ============================================================
# Boman index
# ============================================================

def boman_index(seq: str) -> float:
    """Sum of transfer energies (Boman 2003). High = good AMP candidate."""
    vals = [BOMAN[aa] for aa in seq if aa in BOMAN]
    return float(sum(vals)) if vals else 0.0


# ============================================================
# Aliphatic index (Ikai 1980)
# ============================================================

def aliphatic_index(seq: str) -> float:
    """
    AI = frac(A)*100 + 2.9*frac(V)*100 + 3.9*(frac(I)+frac(L))*100
    High AI → thermostable.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    fa = seq.count("A") / n
    fv = seq.count("V") / n
    fi = seq.count("I") / n
    fl = seq.count("L") / n
    return (fa + 2.9 * fv + 3.9 * (fi + fl)) * 100.0


# ============================================================
# Instability index (Guruprasad et al. 1990)
# ============================================================

def instability_index(seq: str) -> float:
    """
    II = (10 / L) * Σ DIWV(Xi, Xi+1)
    Peptides with II < 40 are predicted stable.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        pair = seq[i] + seq[i + 1]
        total += DIWV.get(pair, 1.0)
    return (10.0 / n) * total


# ============================================================
# Molecular weight
# ============================================================

def molecular_weight(seq: str) -> float:
    """MW of linear peptide (Da): sum of residue weights + H2O."""
    mw = MW_WATER
    for aa in seq:
        mw += AA_MW.get(aa, 0.0) - MW_WATER  # residue = aa − H2O
    return mw


def _residue_mw(seq: str) -> float:
    """Sum of residue (amino acid − H2O) weights, then add one H2O for the chain."""
    # Each residue contributes MW(aa) - 18.015 (peptide bond formation)
    total = sum(AA_MW.get(aa, 0.0) for aa in seq)
    # Full peptide: total - (n-1)*18.015 + 18.015 = total - (n-2)*18.015
    # Simpler: sum(residue_weights) + 18.015
    n = len(seq)
    residue_sum = sum(AA_MW.get(aa, 0.0) - MW_WATER for aa in seq)
    return residue_sum + MW_WATER  # linear peptide


# ============================================================
# AA composition helpers
# ============================================================

POSITIVE_AA = frozenset("KRH")
NEGATIVE_AA = frozenset("DE")
POLAR_AA    = frozenset("NQSTY")
APOLAR_AA   = frozenset("AGVLIPFMWC")
AROMATIC_AA = frozenset("FWY")


def aa_fractions(seq: str) -> Dict[str, float]:
    n = len(seq)
    if n == 0:
        return {k: 0.0 for k in
                ("frac_positive","frac_negative","frac_charged",
                 "frac_polar","frac_apolar","frac_aromatic",
                 "frac_cys","frac_pro","frac_gly")}
    return {
        "frac_positive": sum(1 for aa in seq if aa in POSITIVE_AA) / n,
        "frac_negative": sum(1 for aa in seq if aa in NEGATIVE_AA) / n,
        "frac_charged":  sum(1 for aa in seq if aa in POSITIVE_AA | NEGATIVE_AA) / n,
        "frac_polar":    sum(1 for aa in seq if aa in POLAR_AA)    / n,
        "frac_apolar":   sum(1 for aa in seq if aa in APOLAR_AA)   / n,
        "frac_aromatic": sum(1 for aa in seq if aa in AROMATIC_AA) / n,
        "frac_cys":      seq.count("C") / n,
        "frac_pro":      seq.count("P") / n,
        "frac_gly":      seq.count("G") / n,
    }


# ============================================================
# Per-sequence feature computation
# ============================================================

def compute_qsar(seq: str, ph: float = 7.0) -> Dict[str, float]:
    """
    Compute all QSAR descriptors for a single peptide sequence.
    Non-standard amino acids are silently skipped.
    """
    seq = str(seq).strip().upper()
    # Keep only standard AAs
    seq_clean = "".join(aa for aa in seq if aa in AA_ALPHABET)
    n = len(seq_clean)

    if n == 0:
        # Return NaN-filled dict so downstream code can filter
        keys = [
            "length", "mw", "charge_pH7", "pI", "gravy", "mean_eisenberg",
            "h_moment_helix", "h_moment_beta", "boman_index", "aliphatic_index",
            "instability_index", "frac_positive", "frac_negative", "frac_charged",
            "frac_polar", "frac_apolar", "frac_aromatic", "frac_cys", "frac_pro",
            "frac_gly", "net_charge_ratio", "charge_hydro_ratio",
        ]
        return {k: float("nan") for k in keys}

    q = charge_at_ph7(seq_clean, ph=ph)
    g = gravy(seq_clean)
    fracs = aa_fractions(seq_clean)

    feat = {
        "length":           n,
        "mw":               _residue_mw(seq_clean),
        "charge_pH7":       q,
        "pI":               isoelectric_point(seq_clean),
        "gravy":            g,
        "mean_eisenberg":   mean_eisenberg(seq_clean),
        "h_moment_helix":   hydrophobic_moment(seq_clean, delta_deg=100.0, window=11),
        "h_moment_beta":    hydrophobic_moment(seq_clean, delta_deg=160.0, window=5),
        "boman_index":      boman_index(seq_clean),
        "aliphatic_index":  aliphatic_index(seq_clean),
        "instability_index": instability_index(seq_clean),
        "net_charge_ratio": q / n,
        "charge_hydro_ratio": q / (abs(g) + 1e-3),
    }
    feat.update(fracs)
    return feat


# ============================================================
# Batch processing
# ============================================================

def compute_qsar_batch(
    sequences: List[str],
    ph: float = 7.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute QSAR features for a list of sequences.
    Returns a DataFrame with one row per sequence.
    """
    try:
        from tqdm import tqdm
        it = tqdm(sequences, desc="QSAR", unit="seq", disable=not verbose)
    except ImportError:
        it = sequences

    rows = [compute_qsar(seq, ph=ph) for seq in it]
    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Phase 2A — QSAR feature extraction for peptides"
    )
    p.add_argument("--input", required=True, help="Input CSV with peptide sequences")
    p.add_argument("--output", required=True, help="Output CSV with QSAR features appended")
    p.add_argument(
        "--sequence_col", default="sequence",
        help="Name of the sequence column in the input CSV (default: sequence)"
    )
    p.add_argument(
        "--ph", type=float, default=7.0,
        help="pH for net charge calculation (default: 7.0)"
    )
    p.add_argument(
        "--drop_empty", action="store_true",
        help="Drop rows where the sequence contains no valid amino acids"
    )
    args = p.parse_args()

    print(f"[INFO] Reading: {args.input}")
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    if args.sequence_col not in df.columns:
        # Try lowercase fallback
        lc_cols = {c.lower(): c for c in df.columns}
        if args.sequence_col.lower() in lc_cols:
            args.sequence_col = lc_cols[args.sequence_col.lower()]
        else:
            raise ValueError(
                f"Column '{args.sequence_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    sequences = df[args.sequence_col].astype(str).tolist()
    print(f"[INFO] Computing QSAR for {len(sequences)} sequences at pH {args.ph}")

    qsar_df = compute_qsar_batch(sequences, ph=args.ph, verbose=True)

    # Drop existing QSAR columns to avoid duplicates
    existing_qsar = [c for c in qsar_df.columns if c in df.columns]
    if existing_qsar:
        print(f"[INFO] Overwriting existing columns: {existing_qsar}")
        df = df.drop(columns=existing_qsar)

    out = pd.concat([df.reset_index(drop=True), qsar_df.reset_index(drop=True)], axis=1)

    if args.drop_empty:
        before = len(out)
        out = out.dropna(subset=["length"])
        after = len(out)
        if before != after:
            print(f"[INFO] Dropped {before - after} empty-sequence rows")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[OK] Saved: {args.output} | rows={len(out)} | QSAR cols={len(qsar_df.columns)}")

    # Summary statistics for AMP-relevant features
    print("\n[QSAR summary]")
    summary_cols = ["length", "charge_pH7", "pI", "gravy", "h_moment_helix", "boman_index"]
    for col in summary_cols:
        if col in out.columns:
            vals = out[col].dropna()
            print(
                f"  {col:20s}: mean={vals.mean():.2f}  "
                f"std={vals.std():.2f}  "
                f"[{vals.min():.2f}, {vals.max():.2f}]"
            )


if __name__ == "__main__":
    main()
