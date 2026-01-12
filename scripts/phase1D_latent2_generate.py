#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1D adaptado para checkpoints de phase1C_latent2
Generación + scoring + análisis de diversidad/novelty
"""

import os, re, json, math, random, argparse, glob, itertools, collections
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config comunes ----------
BACTOK = {"ecoli":"<ECOLI>", "kpneumoniae":"<KPN>", "paeruginosa":"<PAER>", "all":"<ALL>"}
AMINO = set(list("ACDEFGHIKLMNPQRSTVWY"))

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------- Carga train para novelty ----------
def load_train_sequences(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "sequence" in df.columns
    seqs = [str(s).strip().upper() for s in df["sequence"].tolist()]
    seqs = ["".join([c for c in s if c in AMINO]) for s in seqs]
    return [s for s in seqs if len(s) > 0]

def kmers(s: str, k: int = 4) -> Set[str]:
    return set(s[i:i+k] for i in range(0, max(0, len(s)-k+1)))

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    inter = len(a & b); uni = len(a | b)
    return inter / (uni + 1e-9)

# ---------- Modelo FT: cabezas reg/cls ----------
class Heads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 1)
    def forward(self, last_hidden):
        mic = self.reg_head(last_hidden).squeeze(-1)
        logit = self.cls_head(last_hidden).squeeze(-1)
        return mic, logit

# ---------- ADAPTADO: Carga para phase1C_latent2 ----------
def load_decoder_and_heads_latent2(ckpt_path: str, base_model: str = "nferruz/ProtGPT2", tokenizer_dir: str = None):
    """
    Carga decoder + heads desde checkpoint Lightning de phase1C_latent2
    """
    
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint no existe: {ckpt_path}")
    
    print(f"[INFO] Cargando checkpoint: {ckpt_path}")
    
    # Cargar checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    
    print(f"[INFO] Checkpoint contiene {len(state_dict)} parámetros")
    
    # Extraer pesos del decoder
    dec_state = {
        k.replace("dec.", ""): v 
        for k, v in state_dict.items() 
        if k.startswith("dec.")
    }
    
    if not dec_state:
        raise ValueError("No se encontraron parámetros del decoder")
    
    # Detectar tamaño del vocabulario del checkpoint
    if "transformer.wte.weight" in dec_state:
        vocab_size_ckpt = dec_state["transformer.wte.weight"].shape[0]
        print(f"[INFO] Vocabulario en checkpoint: {vocab_size_ckpt} tokens")
    else:
        raise ValueError("No se encontró transformer.wte.weight")
    
    # ============================================================
    # TOKENIZER - CARGAR EL GUARDADO
    # ============================================================
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        print(f"[INFO] Cargando tokenizer desde argumento: {tokenizer_dir}")
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)
    else:
        # Buscar en ubicaciones por defecto
        ckpt_dir = os.path.dirname(ckpt_path)
        candidates = [
            os.path.join(ckpt_dir, "tokenizer"),
            os.path.join(os.path.dirname(ckpt_dir), "tokenizer"),
        ]
        
        tok = None
        for cand in candidates:
            if os.path.isdir(cand):
                print(f"[INFO] Cargando tokenizer desde: {cand}")
                tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=False)
                break
        
        if tok is None:
            raise FileNotFoundError(
                f"No se encontró tokenizer. Buscado en:\n" +
                "\n".join(f"  - {c}" for c in candidates) +
                "\nUsa --tokenizer_dir para especificar la ruta."
            )
    
    print(f"[INFO] Tokenizer cargado: {len(tok)} tokens")
    
    # Verificar consistencia
    if len(tok) != vocab_size_ckpt:
        print(f"[WARNING] Mismatch: tokenizer={len(tok)}, checkpoint={vocab_size_ckpt}")
    
    # ============================================================
    # DECODER
    # ============================================================
    print(f"[INFO] Cargando decoder base: {base_model}")
    dec = AutoModelForCausalLM.from_pretrained(
        base_model,
        use_safetensors=True,
        trust_remote_code=False
    )
    
    # Resize al tamaño del CHECKPOINT
    print(f"[INFO] Ajustando embeddings a {vocab_size_ckpt} tokens")
    dec.resize_token_embeddings(vocab_size_ckpt)
    
    # Cargar pesos
    print(f"[INFO] Cargando {len(dec_state)} parámetros del decoder")
    missing, unexpected = dec.load_state_dict(dec_state, strict=False)
    
    if missing:
        print(f"[WARNING] {len(missing)} parámetros faltantes")
    if unexpected:
        print(f"[WARNING] {len(unexpected)} parámetros inesperados")
    
    dec.eval()
    
    # ============================================================
    # HEADS
    # ============================================================
    hid = dec.config.n_embd
    heads = Heads(hid)
    
    reg_state = {k.replace("reg_head.", ""): v for k, v in state_dict.items() if k.startswith("reg_head.")}
    cls_state = {k.replace("cls_head.", ""): v for k, v in state_dict.items() if k.startswith("cls_head.")}
    
    if reg_state and cls_state:
        heads.reg_head.load_state_dict(reg_state)
        heads.cls_head.load_state_dict(cls_state)
        print(f"[INFO] Heads cargados correctamente")
    else:
        print("[WARNING] Heads no encontrados, usando inicialización aleatoria")
    
    heads.eval()
    
    return tok, dec, heads

# ---------- Scoring ----------
@torch.no_grad()
def score_batch(dec, heads, tok, seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = dec.to(device); heads = heads.to(device)
    bos = tok.bos_token or "<BOS>"
    eos = tok.eos_token or "<EOS>"

    logits_all = []
    mics_all = []
    B = 64
    for i in range(0, len(seqs), B):
        batch = seqs[i:i+B]
        ids = [tok.encode(bos + s + eos, add_special_tokens=False) for s in batch]
        L = max(len(x) for x in ids)
        x = np.full((len(ids), L), tok.pad_token_id, dtype=np.int64)
        attn = np.zeros((len(ids), L), dtype=np.int64)
        for r, ids_r in enumerate(ids):
            x[r, :len(ids_r)] = ids_r
            attn[r, :len(ids_r)] = 1
        x = torch.tensor(x, device=device); attn = torch.tensor(attn, device=device)
        out = dec(input_ids=x, attention_mask=attn, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]
        lengths = attn.sum(dim=1).clamp(min=1) - 1
        idx = torch.arange(last_hidden.size(0), device=device)
        h = last_hidden[idx, lengths]
        mic, logit = heads(h)
        mics_all.append(mic.detach().float().cpu().numpy())
        logits_all.append(logit.detach().float().cpu().numpy())

    mics = np.concatenate(mics_all, 0)
    logits = np.concatenate(logits_all, 0)
    probs = 1/(1+np.exp(-logits))
    return mics, probs

# ---------- Sampling ----------
@torch.no_grad()
def sample_candidates(tok, dec, ctrl_token: str, n: int, max_new: int = 64,
                      top_p=0.95, temp=1.0, rep_pen=1.1):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = dec.to(device)

    bos = tok.bos_token or "<BOS>"
    prompt = ctrl_token + bos
    input_ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device=device)

    out = dec.generate(
        input_ids=input_ids,
        do_sample=True, top_p=top_p, temperature=temp,
        repetition_penalty=rep_pen,
        max_new_tokens=max_new, num_return_sequences=n,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
    )

    seqs=[]
    for i in range(out.size(0)):
        text = tok.decode(out[i], skip_special_tokens=False)
        m = re.search(re.escape(bos) + r"(.*?)" + re.escape(tok.eos_token or "<EOS>"), text)
        s = m.group(1) if m else text
        s = "".join([c for c in s if c in AMINO])
        if 5 <= len(s) <= max_new:
            seqs.append(s)

    return seqs

# ---------- Dedup / Metrics ----------
def approx_dedup_by_kmer(seqs: List[str], k=4, thr_j=0.9) -> List[str]:
    kept=[]
    ksets=[]
    for s in seqs:
        ks = kmers(s,k)
        if all(jaccard(ks, ks2) < thr_j for ks2 in ksets):
            kept.append(s); ksets.append(ks)
    return kept

def pick_top_by_scores(seqs, mic, prob, top_k=100):
    idx = np.lexsort((-prob, mic))
    return [(seqs[i], float(mic[i]), float(prob[i])) for i in idx[:top_k]]

# ---------- N-gram KL ----------
def ngram_kl(seqs_gen, seqs_train, n=2):
    def dist(seqs, n):
        cnt = collections.Counter()
        tot = 0
        for s in seqs:
            grams = [s[i:i+n] for i in range(len(s)-n+1)]
            cnt.update(grams); tot += max(len(grams),0)
        V = len(cnt) or 1
        return {k:(v+1)/(tot+V) for k,v in cnt.items()}
    Pg = dist(seqs_gen, n)
    Pt = dist(seqs_train, n)
    keys = set(Pg) | set(Pt)
    kl=0.0
    for k in keys:
        pg = Pg.get(k, 1e-9); pt = Pt.get(k, 1e-9)
        kl += pg * math.log(pg/pt)
    return kl

# ---------- SELF-BLEU proxy ----------
def self_bleu_proxy(seqs, n=4, samples=200):
    if len(seqs) < 2: return 0.0
    rnd = np.random.default_rng(123)
    indices = rnd.choice(len(seqs), size=min(samples, len(seqs)), replace=False)
    scores=[]
    for idx in indices:
        cand = seqs[idx]
        ref  = seqs[rnd.integers(0, len(seqs))]
        def ngrams(s,k): return collections.Counter([s[i:i+k] for i in range(len(s)-k+1)]) if len(s)>=k else collections.Counter()
        w=[]
        for k in range(1, n+1):
            c = ngrams(cand,k); r = ngrams(ref,k)
            inter = sum((c & r).values()); tot = max(sum(c.values()), 1)
            w.append(inter/tot)
        g = math.exp(sum([math.log(max(x,1e-8)) for x in w]) / n)
        scores.append(g)
    return float(np.mean(scores))

# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="CSV de training (para novelty)")
    ap.add_argument("--ckpt_path", required=True, help="Checkpoint .ckpt de phase1C_latent2")
    ap.add_argument("--tokenizer_dir", default=None, help="Directorio del tokenizer (si no está junto al ckpt)")
    ap.add_argument("--decoder_model", default="nferruz/ProtGPT2")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida")
    ap.add_argument("--n_per_bug", type=int, default=1500, help="Candidatos por bacteria")
    ap.add_argument("--max_new",  type=int, default=64, help="Max tokens generados")
    ap.add_argument("--temp", type=float, default=1.0, help="Temperature sampling")
    ap.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling")
    ap.add_argument("--rep_pen", type=float, default=1.1, help="Repetition penalty")
    ap.add_argument("--kmer", type=int, default=4, help="K-mer size para novelty")
    ap.add_argument("--nov_jacc_max", type=float, default=0.6, help="Max Jaccard vs train")
    ap.add_argument("--top_k", type=int, default=100, help="Top K por bacteria")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(42)

    print("="*70)
    print("PHASE 1D - GENERACIÓN CON phase1C_latent2")
    print("="*70)

    # Cargar train
    print(f"\n[INFO] Cargando training set: {args.train_csv}")
    train_seqs = load_train_sequences(args.train_csv)
    train_ksets = [kmers(s, args.kmer) for s in train_seqs]
    print(f"[INFO] Training set: {len(train_seqs)} secuencias")

    # Cargar modelo + heads (ADAPTADO para latent2)
    print(f"\n[INFO] Cargando modelo desde: {args.ckpt_path}")
    tok, dec, heads = load_decoder_and_heads_latent2(
        args.ckpt_path, 
        args.decoder_model,
        tokenizer_dir=args.tokenizer_dir
    )

    results = {}
    all_rows = []

    # ------------------ GENERACIÓN ------------------
    for bug in ["ecoli", "kpneumoniae", "paeruginosa"]:
        print(f"\n{'='*70}")
        print(f"GENERANDO PARA: {bug.upper()}")
        print(f"{'='*70}")
        
        ctrl = BACTOK[bug]

        print(f"[INFO] Generando {args.n_per_bug} candidatos...")
        cand = sample_candidates(tok, dec, ctrl,
                                 n=args.n_per_bug,
                                 max_new=args.max_new,
                                 top_p=args.top_p,
                                 temp=args.temp,
                                 rep_pen=args.rep_pen)

        print(f"[INFO] Candidatos iniciales: {len(cand)}")

        # Limpieza
        cand = ["".join([c for c in s if c in AMINO]) for s in cand]
        cand = [s for s in cand if len(s)>=8]
        print(f"[INFO] Después de filtro mínimo: {len(cand)}")
        
        # Dedup
        cand = approx_dedup_by_kmer(cand, k=args.kmer, thr_j=0.9)
        print(f"[INFO] Después de dedup: {len(cand)}")

        # novelty Jaccard vs train
        print(f"[INFO] Calculando novelty vs training set...")
        def min_jacc_train(s):
            ks = kmers(s, args.kmer)
            step = max(1, len(train_ksets)//2000)
            return min(jaccard(ks, tks) for tks in train_ksets[::step])

        nov = np.array([1.0 - min_jacc_train(s) for s in cand], dtype=np.float32)
        keep = [i for i,s in enumerate(cand) if (1.0 - nov[i]) <= args.nov_jacc_max]
        cand = [cand[i] for i in keep]
        nov = nov[keep]
        
        print(f"[INFO] Después de novelty filter: {len(cand)}")

        if len(cand) == 0:
            print(f"[WARNING] Sin candidatos; relajando umbral y regenerando...")
            cand = sample_candidates(tok, dec, ctrl, n=args.n_per_bug//2,
                                     max_new=args.max_new, top_p=args.top_p,
                                     temp=args.temp, rep_pen=args.rep_pen)
            cand = ["".join([c for c in s if c in AMINO]) for s in cand]
            cand = [s for s in cand if len(s)>=8]

        # Scoring
        print(f"[INFO] Scoring {len(cand)} candidatos...")
        mics, probs = score_batch(dec, heads, tok, cand)

        # Top K
        chosen = pick_top_by_scores(cand, mics, probs, top_k=args.top_k)
        print(f"[INFO] Top-{args.top_k} seleccionados")

        # Métricas
        uniq = len(set(cand)) / max(1, len(cand))
        sbleu = self_bleu_proxy(cand, n=4, samples=min(200, len(cand)))

        d_pair = []
        ck = [kmers(s, args.kmer) for s in cand[:1000]]
        for i in range(0, min(1000, len(ck)), 10):
            for j in range(i+1, min(1000, len(ck)), 50):
                d_pair.append(1.0 - jaccard(ck[i], ck[j]))
        div_mean = float(np.mean(d_pair)) if d_pair else 0.0

        kl1 = ngram_kl(cand, train_seqs, n=1)
        kl2 = ngram_kl(cand, train_seqs, n=2)

        results[bug] = dict(
            n_generated=args.n_per_bug,
            n_after_filters=len(cand),
            uniqueness=round(uniq,4),
            self_bleu_proxy=round(sbleu,4),
            diversity_4mer_mean=round(div_mean,4),
            kl_1gram=round(kl1,4),
            kl_2gram=round(kl2,4),
            mic_median=float(np.median(mics)),
            mic_p10=float(np.percentile(mics,10)),
            mic_p90=float(np.percentile(mics,90)),
            prob_active_mean=float(np.mean(probs)),
        )

        print(f"\n[STATS] {bug}:")
        for k, v in results[bug].items():
            print(f"  {k}: {v}")

        # Guardar
        out_csv = os.path.join(args.out_dir, f"top_{bug}.csv")
        pd.DataFrame([
            {"bug":bug, "sequence":s, "pred_mic":mic, "prob_active":p}
            for (s,mic,p) in chosen
        ]).to_csv(out_csv, index=False)
        print(f"[INFO] Guardado: {out_csv}")

        for (s,mic,p) in chosen:
            all_rows.append((bug,s,mic,p))

    # resumen global
    summary_path = os.path.join(args.out_dir, "gen_summary.csv")
    pd.DataFrame(results).T.to_csv(summary_path)
    print(f"\n[INFO] Resumen guardado: {summary_path}")
    
    all_path = os.path.join(args.out_dir, "top_all_conditions.csv")
    pd.DataFrame(all_rows, columns=["bug","sequence","pred_mic","prob_active"])\
      .to_csv(all_path, index=False)
    print(f"[INFO] Top todos guardado: {all_path}")

    print("\n" + "="*70)
    print("✅ GENERACIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    main()
