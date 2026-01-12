#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1D para LoRA
Generación + scoring + análisis de diversidad/novelty

Diferencia vs phase1D_latent2:
  - Carga modelo con PEFT (LoRA adapter)
  - No usa bridge/prefijo, generación directa condicionada por token bacteria
"""

import os
import re
import json
import math
import random
import argparse
import collections
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Config
# ============================================================

BACTOK = {
    "ecoli": "<ECOLI>",
    "kpneumoniae": "<KPN>",
    "paeruginosa": "<PAER>",
}

AMINO = set("ACDEFGHIKLMNPQRSTVWY")


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilidades
# ============================================================

def kmers(s: str, k: int = 4) -> Set[str]:
    return set(s[i:i+k] for i in range(max(0, len(s) - k + 1)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / (uni + 1e-9)


def clean_sequence(s: str) -> str:
    return "".join(c for c in s.upper() if c in AMINO)


# ============================================================
# Carga de datos
# ============================================================

def load_train_sequences(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "sequence" in df.columns
    seqs = [clean_sequence(str(s)) for s in df["sequence"].tolist()]
    return [s for s in seqs if len(s) > 0]


# ============================================================
# Heads (reg/cls)
# ============================================================

class Heads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, h):
        mic = self.reg_head(h).squeeze(-1)
        logit = self.cls_head(h).squeeze(-1)
        return mic, logit


# ============================================================
# Carga de modelo LoRA
# ============================================================

def load_lora_model(
    lora_adapter_dir: str,
    tokenizer_dir: str,
    model_full_pt: str = None,
    base_model: str = "nferruz/ProtGPT2",
):
    """
    Carga modelo LoRA + heads.
    
    Args:
        lora_adapter_dir: Directorio con el adapter LoRA guardado
        tokenizer_dir: Directorio con el tokenizer
        model_full_pt: Path a model_full.pt (para extraer heads)
        base_model: Modelo base de HuggingFace
    
    Returns:
        tok, dec (con LoRA), heads
    """
    print(f"[INFO] Cargando tokenizer desde: {tokenizer_dir}")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(f"[INFO] Tokenizer: {len(tok)} tokens")
    
    # Cargar modelo base
    print(f"[INFO] Cargando modelo base: {base_model}")
    dec_base = AutoModelForCausalLM.from_pretrained(
        base_model,
        use_safetensors=True,
    )
    
    # Resize embeddings al tamaño del tokenizer guardado
    dec_base.resize_token_embeddings(len(tok))
    
    # Aplicar LoRA adapter
    print(f"[INFO] Cargando LoRA adapter desde: {lora_adapter_dir}")
    dec = PeftModel.from_pretrained(dec_base, lora_adapter_dir)
    dec.eval()
    
    # Contar parámetros
    total = sum(p.numel() for p in dec.parameters())
    trainable = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    print(f"[INFO] Parámetros: {total:,} total, {trainable:,} trainable")
    
    # Hidden size
    hid = dec_base.config.n_embd
    
    # Cargar heads desde model_full.pt
    heads = Heads(hid)
    
    if model_full_pt and os.path.isfile(model_full_pt):
        print(f"[INFO] Cargando heads desde: {model_full_pt}")
        ckpt = torch.load(model_full_pt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
        reg_state = {k.replace("reg_head.", ""): v for k, v in state_dict.items() if k.startswith("reg_head.")}
        cls_state = {k.replace("cls_head.", ""): v for k, v in state_dict.items() if k.startswith("cls_head.")}
        
        if reg_state and cls_state:
            heads.reg_head.load_state_dict(reg_state)
            heads.cls_head.load_state_dict(cls_state)
            print(f"[INFO] Heads cargados correctamente")
        else:
            print("[WARNING] Heads no encontrados en model_full.pt")
    else:
        print("[WARNING] model_full.pt no especificado, heads con inicialización aleatoria")
    
    heads.eval()
    
    return tok, dec, heads


# ============================================================
# Generación
# ============================================================

@torch.no_grad()
def generate_sequences(
    tok,
    dec,
    ctrl_token: str,
    n: int,
    max_new: int = 64,
    top_p: float = 0.95,
    temp: float = 1.0,
    rep_pen: float = 1.1,
    batch_size: int = 50,
) -> List[str]:
    """Genera secuencias condicionadas por token de bacteria."""
    
    device = next(dec.parameters()).device
    
    bos = tok.bos_token or "<BOS>"
    eos = tok.eos_token or "<EOS>"
    
    prompt = ctrl_token + bos
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    
    all_seqs = []
    
    for i in range(0, n, batch_size):
        batch_n = min(batch_size, n - i)
        
        input_ids = torch.tensor([prompt_ids] * batch_n, device=device)
        
        outputs = dec.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=top_p,
            temperature=temp,
            repetition_penalty=rep_pen,
            max_new_tokens=max_new,
            num_return_sequences=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        
        for j in range(outputs.size(0)):
            text = tok.decode(outputs[j], skip_special_tokens=False)
            
            # Extraer secuencia entre BOS y EOS
            pattern = re.escape(bos) + r"(.*?)" + re.escape(eos)
            match = re.search(pattern, text)
            
            if match:
                seq = match.group(1)
            else:
                # Fallback: todo después de BOS
                idx = text.find(bos)
                seq = text[idx + len(bos):] if idx >= 0 else text
            
            seq = clean_sequence(seq)
            
            if len(seq) >= 5:
                all_seqs.append(seq)
    
    return all_seqs


# ============================================================
# Scoring
# ============================================================

@torch.no_grad()
def score_sequences(
    tok,
    dec,
    heads,
    sequences: List[str],
    bug: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula MIC y prob_active para cada secuencia.
    Usa el token de bacteria correspondiente.
    """
    device = next(dec.parameters()).device
    heads = heads.to(device)
    
    bos = tok.bos_token or "<BOS>"
    eos = tok.eos_token or "<EOS>"
    ctrl_token = BACTOK[bug]
    
    all_mics = []
    all_probs = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        
        # Tokenizar: <BUG><BOS>seq<EOS>
        texts = [ctrl_token + bos + s + eos for s in batch_seqs]
        ids_list = [tok.encode(t, add_special_tokens=False) for t in texts]
        
        max_len = max(len(ids) for ids in ids_list)
        
        input_ids = torch.full((len(ids_list), max_len), tok.pad_token_id, device=device)
        attn_mask = torch.zeros((len(ids_list), max_len), device=device)
        
        for j, ids in enumerate(ids_list):
            input_ids[j, :len(ids)] = torch.tensor(ids, device=device)
            attn_mask[j, :len(ids)] = 1
        
        # Forward
        outputs = dec(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # Extraer h_T (último token con atención)
        last_hidden = outputs.hidden_states[-1]
        lengths = attn_mask.sum(dim=1).long() - 1
        idx = torch.arange(len(ids_list), device=device)
        h_T = last_hidden[idx, lengths]
        
        # Heads
        mic, logit = heads(h_T)
        prob = torch.sigmoid(logit)
        
        all_mics.append(mic.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
    
    return np.concatenate(all_mics), np.concatenate(all_probs)


# ============================================================
# Métricas
# ============================================================

def compute_diversity_metrics(
    generated: List[str],
    train_seqs: List[str],
    k: int = 4,
) -> Dict[str, float]:
    """Calcula métricas de diversidad y novelty."""
    
    if len(generated) == 0:
        return {
            "n_generated": 0,
            "n_unique": 0,
            "uniqueness": 0.0,
            "diversity_4mer": 0.0,
            "novelty_mean": 0.0,
            "novelty_min": 0.0,
        }
    
    # Uniqueness
    unique = set(generated)
    uniqueness = len(unique) / len(generated)
    
    # Diversidad
    gen_kmers = [kmers(s, k) for s in generated[:500]]
    distances = []
    for i in range(min(len(gen_kmers), 100)):
        for j in range(i + 1, min(len(gen_kmers), 100)):
            distances.append(1.0 - jaccard(gen_kmers[i], gen_kmers[j]))
    diversity = np.mean(distances) if distances else 0.0
    
    # Novelty vs training
    train_sample = train_seqs[::max(1, len(train_seqs)//1000)]
    train_kmers = [kmers(s, k) for s in train_sample]
    
    novelties = []
    for g_km in gen_kmers[:200]:
        if train_kmers:
            min_sim = min(jaccard(g_km, t_km) for t_km in train_kmers)
            novelties.append(1.0 - min_sim)
    
    return {
        "n_generated": len(generated),
        "n_unique": len(unique),
        "uniqueness": round(uniqueness, 4),
        "diversity_4mer": round(diversity, 4),
        "novelty_mean": round(np.mean(novelties), 4) if novelties else 0.0,
        "novelty_min": round(np.min(novelties), 4) if novelties else 0.0,
    }


def ngram_kl(seqs_gen: List[str], seqs_train: List[str], n: int = 2) -> float:
    """KL divergence de n-gramas."""
    def dist(seqs, n):
        cnt = collections.Counter()
        tot = 0
        for s in seqs:
            grams = [s[i:i+n] for i in range(len(s) - n + 1)]
            cnt.update(grams)
            tot += len(grams)
        V = len(cnt) or 1
        return {k: (v + 1) / (tot + V) for k, v in cnt.items()}
    
    Pg = dist(seqs_gen, n)
    Pt = dist(seqs_train, n)
    keys = set(Pg) | set(Pt)
    
    kl = 0.0
    for k in keys:
        pg = Pg.get(k, 1e-9)
        pt = Pt.get(k, 1e-9)
        kl += pg * math.log(pg / pt)
    
    return round(kl, 4)


# ============================================================
# Filtrado
# ============================================================

def filter_and_rank(
    sequences: List[str],
    mics: np.ndarray,
    probs: np.ndarray,
    train_seqs: List[str],
    min_len: int = 8,
    max_len: int = 64,
    max_jaccard: float = 0.6,
    min_prob: float = 0.0,
    k: int = 4,
    top_k: int = 100,
) -> List[Tuple[str, float, float, float]]:
    """Filtra y rankea secuencias."""
    
    train_sample = train_seqs[::max(1, len(train_seqs)//500)]
    train_kmers = [kmers(s, k) for s in train_sample]
    
    results = []
    
    for seq, mic, prob in zip(sequences, mics, probs):
        # Filtros
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        if prob < min_prob:
            continue
        
        # Novelty
        seq_km = kmers(seq, k)
        if train_kmers:
            min_jacc = min(jaccard(seq_km, t_km) for t_km in train_kmers)
        else:
            min_jacc = 0.0
        
        novelty = 1.0 - min_jacc
        
        if min_jacc > max_jaccard:
            continue
        
        results.append((seq, float(mic), float(prob), float(novelty)))
    
    # Rankear por MIC bajo y prob alta
    results.sort(key=lambda x: (x[1], -x[2]))
    
    return results[:top_k]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1D - Generación con LoRA")
    
    # Paths
    parser.add_argument("--train_csv", required=True, help="CSV de training")
    parser.add_argument("--lora_adapter_dir", required=True, help="Directorio del LoRA adapter")
    parser.add_argument("--tokenizer_dir", required=True, help="Directorio del tokenizer")
    parser.add_argument("--model_full_pt", required=True, help="Path a model_full.pt (para heads)")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida")
    
    # Generación
    parser.add_argument("--n_per_bug", type=int, default=500)
    parser.add_argument("--max_new", type=int, default=64)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rep_pen", type=float, default=1.1)
    
    # Filtrado
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_jaccard", type=float, default=0.6)
    parser.add_argument("--min_prob", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=100)
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    print("=" * 70)
    print("PHASE 1D - GENERACIÓN CON LoRA")
    print("=" * 70)
    
    # Cargar training
    print(f"\n[INFO] Cargando training set: {args.train_csv}")
    train_seqs = load_train_sequences(args.train_csv)
    print(f"[INFO] Training: {len(train_seqs)} secuencias")
    
    # Cargar modelo
    print(f"\n[INFO] Cargando modelo LoRA...")
    tok, dec, heads = load_lora_model(
        lora_adapter_dir=args.lora_adapter_dir,
        tokenizer_dir=args.tokenizer_dir,
        model_full_pt=args.model_full_pt,
    )
    dec = dec.to(device)
    heads = heads.to(device)
    
    # Generación por bacteria
    all_results = {}
    all_rows = []
    
    for bug in ["ecoli", "kpneumoniae", "paeruginosa"]:
        print(f"\n{'=' * 70}")
        print(f"GENERANDO PARA: {bug.upper()}")
        print(f"{'=' * 70}")
        
        ctrl_token = BACTOK[bug]
        
        # Generar
        print(f"[INFO] Generando {args.n_per_bug} secuencias...")
        generated = generate_sequences(
            tok=tok,
            dec=dec,
            ctrl_token=ctrl_token,
            n=args.n_per_bug,
            max_new=args.max_new,
            top_p=args.top_p,
            temp=args.temp,
            rep_pen=args.rep_pen,
        )
        
        # Filtrar vacías
        generated = [s for s in generated if len(s) >= args.min_len]
        print(f"[INFO] Secuencias válidas: {len(generated)}")
        
        if len(generated) == 0:
            print(f"[WARNING] Sin secuencias para {bug}")
            continue
        
        # Scoring
        print(f"[INFO] Scoring {len(generated)} secuencias (bug={bug})...")
        mics, probs = score_sequences(
            tok=tok,
            dec=dec,
            heads=heads,
            sequences=generated,
            bug=bug,
        )
        
        # Filtrar y rankear
        print(f"[INFO] Filtrando y rankeando...")
        top_results = filter_and_rank(
            sequences=generated,
            mics=mics,
            probs=probs,
            train_seqs=train_seqs,
            min_len=args.min_len,
            max_len=args.max_new,
            max_jaccard=args.max_jaccard,
            min_prob=args.min_prob,
            top_k=args.top_k,
        )
        
        print(f"[INFO] Top-{args.top_k} después de filtrado: {len(top_results)}")
        
        # Métricas
        diversity_metrics = compute_diversity_metrics(generated, train_seqs)
        kl1 = ngram_kl(generated, train_seqs, n=1)
        kl2 = ngram_kl(generated, train_seqs, n=2)
        
        all_results[bug] = {
            **diversity_metrics,
            "kl_1gram": kl1,
            "kl_2gram": kl2,
            "mic_median": float(np.median(mics)),
            "mic_p10": float(np.percentile(mics, 10)),
            "mic_p90": float(np.percentile(mics, 90)),
            "prob_active_mean": float(np.mean(probs)),
            "n_top_filtered": len(top_results),
        }
        
        print(f"\n[STATS] {bug}:")
        for k, v in all_results[bug].items():
            print(f"  {k}: {v}")
        
        # Guardar
        out_csv = os.path.join(args.out_dir, f"top_{bug}.csv")
        df = pd.DataFrame([
            {"bug": bug, "sequence": s, "pred_mic": mic, "prob_active": prob, "novelty": nov}
            for (s, mic, prob, nov) in top_results
        ])
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Guardado: {out_csv}")
        
        for (s, mic, prob, nov) in top_results:
            all_rows.append((bug, s, mic, prob, nov))
    
    # Resumen
    summary_path = os.path.join(args.out_dir, "gen_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\n[INFO] Resumen guardado: {summary_path}")
    
    all_path = os.path.join(args.out_dir, "top_all.csv")
    pd.DataFrame(all_rows, columns=["bug", "sequence", "pred_mic", "prob_active", "novelty"])\
      .to_csv(all_path, index=False)
    print(f"[INFO] Top todos guardado: {all_path}")
    
    metrics_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    pd.DataFrame(all_results).T.to_csv(metrics_csv)
    print(f"[INFO] Métricas guardado: {metrics_csv}")
    
    print("\n" + "=" * 70)
    print("✅ GENERACIÓN COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
