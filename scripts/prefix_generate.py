#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1D para Prefix-tuning (v2 - CORREGIDO)
Generación condicionada por z → prefijo → ProtGPT2

Correcciones respecto a v1:
  - Scoring CON prefijo (consistente con training)
  - Manejo correcto de attention mask en generación

Estrategias de generación:
  - seed:    Usar z de péptidos activos conocidos directamente
  - sample:  Muestrear z ~ N(μ, Σ) de la distribución de péptidos activos
  - perturb: z_new = z_seed + ε, donde ε ~ N(0, σ)

Uso:
  python phase1D_prefix_v2.py \
    --train_csv data/training_set.csv \
    --encoder_ckpt runs/phase1A_v2/artifacts/.../encoder_state_dict.pt \
    --prefix_ckpt runs/phase1C_prefix_v3/artifacts/prefix_mlp.pt \
    --heads_ckpt runs/phase1C_prefix_v3/artifacts/heads.pt \
    --tokenizer_dir runs/phase1C_prefix_v3/artifacts/tokenizer \
    --out_dir results/gen_prefix \
    --strategy perturb \
    --perturb_sigma 0.5
"""

import os
import sys
import re
import json
import math
import random
import argparse
import collections
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuración
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
    """Extrae k-mers de una secuencia."""
    return set(s[i:i+k] for i in range(max(0, len(s) - k + 1)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Similitud Jaccard entre dos conjuntos."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / (uni + 1e-9)


def clean_sequence(s: str) -> str:
    """Limpia una secuencia dejando solo aminoácidos válidos."""
    return "".join(c for c in s.upper() if c in AMINO)


# ============================================================
# Carga de datos
# ============================================================

def load_training_data(csv_path: str, mic_threshold: float = 4.0) -> Tuple[Dict, List[str]]:
    """
    Carga datos de training y filtra péptidos activos por bacteria.
    
    Returns:
        Dict[bacteria] -> List[(sequence, mic)]
        List[str] - todas las secuencias de training
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    assert "sequence" in df.columns, "CSV debe tener columna 'sequence'"
    
    bugs = [b for b in BACTOK.keys() if b in df.columns]
    print(f"[INFO] Bacterias encontradas: {bugs}")
    
    data = {bug: [] for bug in bugs}
    all_seqs = []
    
    for _, row in df.iterrows():
        seq = clean_sequence(str(row["sequence"]))
        if len(seq) < 5:
            continue
        
        all_seqs.append(seq)
        
        for bug in bugs:
            val = row[bug]
            if pd.isna(val):
                continue
            try:
                mic = float(val)
                if mic < mic_threshold:  # Activo
                    data[bug].append((seq, mic))
            except (ValueError, TypeError):
                continue
    
    for bug in bugs:
        print(f"[INFO] {bug}: {len(data[bug])} péptidos activos (MIC < {mic_threshold})")
    
    return data, all_seqs


# ============================================================
# Carga de modelos
# ============================================================

def load_encoder_1A(ckpt_path: str, codes_dir: str):
    """Carga el encoder 1A desde checkpoint."""
    # Asegurar que podemos importar phase1A_v2
    if codes_dir not in sys.path:
        sys.path.insert(0, codes_dir)
    
    from phase1A_v2 import EncoderVAE
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    h = ckpt["hparams"]
    
    enc = EncoderVAE(latent_dim=int(h["latent_dim"]))
    enc.load_state_dict(ckpt["state_dict"], strict=True)
    enc.eval()
    
    for p in enc.parameters():
        p.requires_grad = False
    
    return enc, int(h["max_len"]), int(h["latent_dim"])


def load_prefix_mlp(ckpt_path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    """Carga el MLP del prefijo."""
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt["hparams"]
    
    zdim = hparams["zdim"]
    prefix_len = hparams["prefix_len"]
    hid = hparams["hid"]
    
    # Recrear arquitectura exacta de phase1C_prefix
    prefix_mlp = nn.Sequential(
        nn.Linear(zdim, zdim * 4),
        nn.LayerNorm(zdim * 4),
        nn.GELU(),
        nn.Linear(zdim * 4, prefix_len * hid),
    )
    
    prefix_mlp.load_state_dict(ckpt["state_dict"])
    prefix_mlp.eval()
    
    for p in prefix_mlp.parameters():
        p.requires_grad = False
    
    return prefix_mlp, hparams


def load_heads(ckpt_path: str, hidden_size: int, device: str = "cpu") -> Tuple[nn.Module, nn.Module]:
    """Carga los heads de regresión y clasificación."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    reg_head = nn.Linear(hidden_size, 1)
    cls_head = nn.Linear(hidden_size, 1)
    
    reg_head.load_state_dict(ckpt["reg_head"])
    cls_head.load_state_dict(ckpt["cls_head"])
    
    reg_head.eval()
    cls_head.eval()
    
    for p in reg_head.parameters():
        p.requires_grad = False
    for p in cls_head.parameters():
        p.requires_grad = False
    
    return reg_head, cls_head


# ============================================================
# Estrategias de generación de z
# ============================================================

def encode_sequences(encoder, sequences: List[str], max_len: int, device: str) -> torch.Tensor:
    """Codifica secuencias usando el encoder 1A."""
    z, _ = encoder.encode_sequences(sequences, max_len=max_len, device=device, use_mu=True)
    return z


def strategy_seed(
    z_active: torch.Tensor,
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Estrategia SEED: Usar z de péptidos activos directamente.
    Selecciona aleatoriamente n_samples de z_active.
    """
    n_active = z_active.size(0)
    indices = rng.choice(n_active, size=min(n_samples, n_active), replace=(n_samples > n_active))
    return z_active[indices]


def strategy_sample(
    z_active: torch.Tensor,
    n_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Estrategia SAMPLE: Muestrear z ~ N(μ, Σ) de la distribución de activos.
    """
    z_np = z_active.cpu().numpy()
    
    mu = z_np.mean(axis=0)
    
    # Covarianza con regularización
    cov = np.cov(z_np.T)
    cov = cov + 1e-4 * np.eye(cov.shape[0])
    
    # Muestrear
    samples = rng.multivariate_normal(mu, cov, size=n_samples)
    
    return torch.tensor(samples, dtype=z_active.dtype)


def strategy_perturb(
    z_active: torch.Tensor,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Estrategia PERTURB: z_new = z_seed + ε, donde ε ~ N(0, σ²I)
    """
    n_active = z_active.size(0)
    zdim = z_active.size(1)
    
    # Seleccionar semillas
    indices = rng.choice(n_active, size=n_samples, replace=(n_samples > n_active))
    z_seeds = z_active[indices].clone()
    
    # Añadir ruido
    noise = torch.randn(n_samples, zdim) * sigma
    z_perturbed = z_seeds + noise.to(z_seeds.device)
    
    return z_perturbed


# ============================================================
# Generación con prefijo
# ============================================================

@torch.no_grad()
def generate_with_prefix(
    tok,
    dec,
    prefix_mlp,
    z: torch.Tensor,
    prefix_len: int,
    hid: int,
    ctrl_token: str,
    max_new: int = 64,
    top_p: float = 0.95,
    temp: float = 1.0,
    rep_pen: float = 1.1,
    batch_size: int = 32,
) -> List[str]:
    """
    Genera secuencias usando z → prefijo → decoder.
    
    Returns:
        List[str]: Secuencias generadas (limpias, solo aminoácidos)
    """
    device = z.device
    dec = dec.to(device)
    prefix_mlp = prefix_mlp.to(device)
    
    bos = tok.bos_token or "<BOS>"
    eos = tok.eos_token or "<EOS>"
    eos_id = tok.eos_token_id
    
    all_seqs = []
    
    for i in range(0, z.size(0), batch_size):
        z_batch = z[i:i+batch_size]
        B = z_batch.size(0)
        
        # z → prefijo
        prefix_flat = prefix_mlp(z_batch)
        prefix_emb = prefix_flat.view(B, prefix_len, hid)
        
        # Prompt: token bacteria + BOS
        prompt = ctrl_token + bos
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids] * B, device=device)
        
        # Embeddings del prompt
        prompt_emb = dec.get_input_embeddings()(prompt_tensor)
        
        # Concatenar prefijo + prompt
        inputs_embeds = torch.cat([prefix_emb, prompt_emb], dim=1)
        
        # Attention mask inicial
        seq_len = prefix_len + len(prompt_ids)
        attn_mask = torch.ones(B, seq_len, device=device)
        
        # Generar token por token
        generated_ids = []
        past_key_values = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_new):
            if step == 0:
                # Primera iteración: usar inputs_embeds
                outputs = dec(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Iteraciones siguientes: usar el último token generado
                last_token = generated_ids[-1].unsqueeze(-1)
                outputs = dec(
                    input_ids=last_token,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [B, vocab]
            
            # Aplicar temperatura
            logits = logits / temp
            
            # Repetition penalty
            if rep_pen != 1.0 and len(generated_ids) > 0:
                for b in range(B):
                    seen_tokens = set()
                    for prev in generated_ids:
                        seen_tokens.add(prev[b].item())
                    for token_id in seen_tokens:
                        logits[b, token_id] /= rep_pen
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remover tokens fuera de top-p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            for b in range(B):
                indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = float('-inf')
            
            # Samplear
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Marcar como terminado si es EOS
            finished = finished | (next_token == eos_id)
            
            generated_ids.append(next_token)
            
            # Actualizar attention mask
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, device=device)], dim=1)
            
            # Terminar si todos han generado EOS
            if finished.all():
                break
        
        # Decodificar
        if generated_ids:
            generated_tensor = torch.stack(generated_ids, dim=1)  # [B, seq_len]
            for b in range(B):
                # Encontrar EOS
                seq_ids = generated_tensor[b].tolist()
                if eos_id in seq_ids:
                    eos_pos = seq_ids.index(eos_id)
                    seq_ids = seq_ids[:eos_pos]
                
                text = tok.decode(seq_ids, skip_special_tokens=True)
                seq = clean_sequence(text)
                all_seqs.append(seq)
        else:
            all_seqs.extend([""] * B)
    
    return all_seqs


# ============================================================
# Scoring CON PREFIJO (CRÍTICO - consistente con training)
# ============================================================

@torch.no_grad()
def score_sequences_with_prefix(
    tok,
    dec,
    prefix_mlp,
    encoder,
    reg_head,
    cls_head,
    sequences: List[str],
    enc_max_len: int,
    prefix_len: int,
    hid: int,
    bug: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula MIC predicho y probabilidad de actividad para cada secuencia.
    
    IMPORTANTE: Usa el prefijo derivado de la secuencia, igual que durante training.
    
    Args:
        bug: Bacteria target ("ecoli", "kpneumoniae", "paeruginosa")
    
    Flujo:
        secuencia → encoder → z → prefix_mlp → prefijo
        prefijo + <BUG> + <BOS> + secuencia + <EOS> → decoder → h_T
        h_T → heads → MIC, prob_active
    """
    device = next(dec.parameters()).device
    
    bos = tok.bos_token or "<BOS>"
    eos = tok.eos_token or "<EOS>"
    
    # Usar el token de bacteria correcto (consistente con generación)
    ctrl_token = BACTOK[bug]
    
    all_mics = []
    all_probs = []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        B = len(batch_seqs)
        
        # 1. Codificar secuencias → z
        z, _ = encoder.encode_sequences(batch_seqs, max_len=enc_max_len, device=device, use_mu=True)
        
        # 2. z → prefijo
        prefix_flat = prefix_mlp(z)
        prefix_emb = prefix_flat.view(B, prefix_len, hid)
        
        # 3. Tokenizar secuencias: <BUG><BOS>seq<EOS>
        texts = [ctrl_token + bos + s + eos for s in batch_seqs]
        ids_list = [tok.encode(t, add_special_tokens=False) for t in texts]
        
        max_seq_len = max(len(ids) for ids in ids_list)
        
        input_ids = torch.full((B, max_seq_len), tok.pad_token_id, device=device)
        token_attn = torch.zeros((B, max_seq_len), device=device)
        
        for j, ids in enumerate(ids_list):
            input_ids[j, :len(ids)] = torch.tensor(ids, device=device)
            token_attn[j, :len(ids)] = 1
        
        # 4. Embeddings de tokens
        token_emb = dec.get_input_embeddings()(input_ids)
        
        # 5. Concatenar prefijo + tokens
        inputs_embeds = torch.cat([prefix_emb, token_emb], dim=1)
        
        # 6. Attention mask extendida
        prefix_attn = torch.ones(B, prefix_len, device=device)
        attn_extended = torch.cat([prefix_attn, token_attn], dim=1)
        
        # 7. Forward
        outputs = dec(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_extended,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # 8. Extraer h_T (último token con atención, ajustado por prefijo)
        last_hidden = outputs.hidden_states[-1]
        lengths = token_attn.sum(dim=1).long() - 1 + prefix_len  # Ajustar por prefijo
        idx = torch.arange(B, device=device)
        h_T = last_hidden[idx, lengths]
        
        # 9. Predicciones
        mic = reg_head(h_T).squeeze(-1)
        logit = cls_head(h_T).squeeze(-1)
        prob = torch.sigmoid(logit)
        
        all_mics.append(mic.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
    
    return np.concatenate(all_mics), np.concatenate(all_probs)


# ============================================================
# Métricas de diversidad
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
            "n_valid": 0,
            "uniqueness": 0.0,
            "diversity_4mer": 0.0,
            "novelty_mean": 0.0,
            "novelty_min": 0.0,
        }
    
    # Uniqueness
    unique = set(generated)
    uniqueness = len(unique) / len(generated)
    
    # Diversidad: distancia media entre pares
    gen_kmers = [kmers(s, k) for s in generated[:500]]
    distances = []
    for i in range(0, min(len(gen_kmers), 100)):
        for j in range(i + 1, min(len(gen_kmers), 100)):
            distances.append(1.0 - jaccard(gen_kmers[i], gen_kmers[j]))
    diversity = np.mean(distances) if distances else 0.0
    
    # Novelty vs training (muestrear para eficiencia)
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
    """Calcula KL divergence de n-gramas."""
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
# Filtrado y selección
# ============================================================

def filter_and_rank(
    sequences: List[str],
    mics: np.ndarray,
    probs: np.ndarray,
    train_seqs: List[str],
    min_len: int = 8,
    max_len: int = 64,
    max_jaccard: float = 0.6,
    k: int = 4,
    top_k: int = 100,
) -> List[Tuple[str, float, float, float]]:
    """
    Filtra y rankea secuencias generadas.
    
    Returns:
        List[(sequence, mic, prob, novelty)]
    """
    # Muestrear training para eficiencia
    train_sample = train_seqs[::max(1, len(train_seqs)//500)]
    train_kmers = [kmers(s, k) for s in train_sample]
    
    results = []
    
    for seq, mic, prob in zip(sequences, mics, probs):
        # Filtros básicos
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        # Novelty
        seq_km = kmers(seq, k)
        if train_kmers:
            min_jacc = min(jaccard(seq_km, t_km) for t_km in train_kmers)
        else:
            min_jacc = 0.0
        novelty = 1.0 - min_jacc
        
        if min_jacc > max_jaccard:
            continue  # Muy similar a training
        
        results.append((seq, float(mic), float(prob), float(novelty)))
    
    # Rankear por MIC bajo y prob alta
    results.sort(key=lambda x: (x[1], -x[2]))
    
    return results[:top_k]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1D - Generación con Prefix-tuning (v2)")
    
    # Paths
    parser.add_argument("--train_csv", required=True, help="CSV de training")
    parser.add_argument("--encoder_ckpt", required=True, help="Checkpoint del encoder 1A")
    parser.add_argument("--prefix_ckpt", required=True, help="Checkpoint de prefix_mlp.pt")
    parser.add_argument("--heads_ckpt", required=True, help="Checkpoint de heads.pt")
    parser.add_argument("--tokenizer_dir", required=True, help="Directorio del tokenizer")
    parser.add_argument("--codes_dir", required=True, help="Directorio con phase1A_v2.py")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida")
    
    # Estrategia
    parser.add_argument("--strategy", choices=["seed", "sample", "perturb"], default="perturb",
                        help="Estrategia de generación de z")
    parser.add_argument("--perturb_sigma", type=float, default=0.5,
                        help="Sigma para estrategia perturb")
    
    # Generación
    parser.add_argument("--n_per_bug", type=int, default=500, help="Candidatos por bacteria")
    parser.add_argument("--max_new", type=int, default=64, help="Max tokens generados")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling")
    parser.add_argument("--rep_pen", type=float, default=1.1, help="Repetition penalty")
    
    # Filtrado
    parser.add_argument("--min_len", type=int, default=8, help="Longitud mínima")
    parser.add_argument("--max_jaccard", type=float, default=0.6, help="Max Jaccard vs train")
    parser.add_argument("--top_k", type=int, default=100, help="Top K por bacteria")
    
    # MIC threshold
    parser.add_argument("--mic_threshold", type=float, default=4.0,
                        help="MIC threshold para considerar péptido activo")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    print("=" * 70)
    print("PHASE 1D - GENERACIÓN CON PREFIX-TUNING (v2 CORREGIDO)")
    print(f"Estrategia: {args.strategy.upper()}")
    print("=" * 70)
    
    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------
    print(f"\n[INFO] Cargando training set: {args.train_csv}")
    active_data, all_train_seqs = load_training_data(args.train_csv, args.mic_threshold)
    
    # --------------------------------------------------------
    # Cargar modelos
    # --------------------------------------------------------
    print(f"\n[INFO] Cargando encoder 1A: {args.encoder_ckpt}")
    encoder, enc_max_len, zdim = load_encoder_1A(args.encoder_ckpt, args.codes_dir)
    encoder = encoder.to(device)
    print(f"[INFO] Encoder: zdim={zdim}, max_len={enc_max_len}")
    
    print(f"\n[INFO] Cargando prefix MLP: {args.prefix_ckpt}")
    prefix_mlp, prefix_hparams = load_prefix_mlp(args.prefix_ckpt, device)
    prefix_mlp = prefix_mlp.to(device)
    prefix_len = prefix_hparams["prefix_len"]
    hid = prefix_hparams["hid"]
    print(f"[INFO] Prefix: len={prefix_len}, hid={hid}")
    
    print(f"\n[INFO] Cargando tokenizer: {args.tokenizer_dir}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"[INFO] Tokenizer: {len(tok)} tokens")
    
    print(f"\n[INFO] Cargando ProtGPT2 base (congelado)")
    dec = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2", use_safetensors=True)
    dec.resize_token_embeddings(len(tok))  # Resize con tokenizer guardado
    dec = dec.to(device)
    dec.eval()
    
    print(f"\n[INFO] Cargando heads: {args.heads_ckpt}")
    reg_head, cls_head = load_heads(args.heads_ckpt, hid, device)
    reg_head = reg_head.to(device)
    cls_head = cls_head.to(device)
    
    # --------------------------------------------------------
    # Generación por bacteria
    # --------------------------------------------------------
    rng = np.random.default_rng(42)
    all_results = {}
    all_rows = []
    
    for bug in ["ecoli", "kpneumoniae", "paeruginosa"]:
        print(f"\n{'=' * 70}")
        print(f"GENERANDO PARA: {bug.upper()}")
        print(f"{'=' * 70}")
        
        ctrl_token = BACTOK[bug]
        
        # Obtener péptidos activos para esta bacteria
        if bug not in active_data or len(active_data[bug]) == 0:
            print(f"[WARNING] No hay péptidos activos para {bug}, saltando...")
            continue
        
        active_seqs = [seq for seq, mic in active_data[bug]]
        print(f"[INFO] Péptidos activos disponibles: {len(active_seqs)}")
        
        # Codificar péptidos activos → z
        print(f"[INFO] Codificando péptidos activos...")
        z_active = encode_sequences(encoder, active_seqs, enc_max_len, device)
        print(f"[INFO] z_active shape: {z_active.shape}")
        
        # Generar z según estrategia
        print(f"[INFO] Generando z con estrategia: {args.strategy}")
        
        if args.strategy == "seed":
            z_gen = strategy_seed(z_active, args.n_per_bug, rng)
        elif args.strategy == "sample":
            z_gen = strategy_sample(z_active, args.n_per_bug, rng)
        elif args.strategy == "perturb":
            z_gen = strategy_perturb(z_active, args.n_per_bug, args.perturb_sigma, rng)
        else:
            raise ValueError(f"Estrategia desconocida: {args.strategy}")
        
        z_gen = z_gen.to(device)
        print(f"[INFO] z_gen shape: {z_gen.shape}")
        
        # Generar secuencias
        print(f"[INFO] Generando {z_gen.size(0)} secuencias...")
        generated = generate_with_prefix(
            tok=tok,
            dec=dec,
            prefix_mlp=prefix_mlp,
            z=z_gen,
            prefix_len=prefix_len,
            hid=hid,
            ctrl_token=ctrl_token,
            max_new=args.max_new,
            top_p=args.top_p,
            temp=args.temp,
            rep_pen=args.rep_pen,
        )
        
        # Filtrar secuencias vacías
        generated = [s for s in generated if len(s) >= args.min_len]
        print(f"[INFO] Secuencias válidas: {len(generated)}")
        
        if len(generated) == 0:
            print(f"[WARNING] Sin secuencias válidas para {bug}")
            continue
        
        # Scoring CON PREFIJO (CORREGIDO)
        print(f"[INFO] Scoring {len(generated)} secuencias (con prefijo, bug={bug})...")
        mics, probs = score_sequences_with_prefix(
            tok=tok,
            dec=dec,
            prefix_mlp=prefix_mlp,
            encoder=encoder,
            reg_head=reg_head,
            cls_head=cls_head,
            sequences=generated,
            enc_max_len=enc_max_len,
            prefix_len=prefix_len,
            hid=hid,
            bug=bug,
        )
        
        # Filtrar y rankear
        print(f"[INFO] Filtrando y rankeando...")
        top_results = filter_and_rank(
            sequences=generated,
            mics=mics,
            probs=probs,
            train_seqs=all_train_seqs,
            min_len=args.min_len,
            max_len=args.max_new,
            max_jaccard=args.max_jaccard,
            top_k=args.top_k,
        )
        
        print(f"[INFO] Top-{args.top_k} después de filtrado: {len(top_results)}")
        
        # Métricas
        diversity_metrics = compute_diversity_metrics(generated, all_train_seqs)
        kl1 = ngram_kl(generated, all_train_seqs, n=1)
        kl2 = ngram_kl(generated, all_train_seqs, n=2)
        
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
        
        # Guardar top por bacteria
        out_csv = os.path.join(args.out_dir, f"top_{bug}.csv")
        df = pd.DataFrame([
            {"bug": bug, "sequence": s, "pred_mic": mic, "prob_active": prob, "novelty": nov}
            for (s, mic, prob, nov) in top_results
        ])
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Guardado: {out_csv}")
        
        for (s, mic, prob, nov) in top_results:
            all_rows.append((bug, s, mic, prob, nov))
    
    # --------------------------------------------------------
    # Resumen global
    # --------------------------------------------------------
    summary_path = os.path.join(args.out_dir, "gen_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "strategy": args.strategy,
            "perturb_sigma": args.perturb_sigma if args.strategy == "perturb" else None,
            "n_per_bug": args.n_per_bug,
            "results": all_results,
        }, f, indent=2)
    print(f"\n[INFO] Resumen guardado: {summary_path}")
    
    # CSV con todos los resultados
    all_path = os.path.join(args.out_dir, "top_all.csv")
    pd.DataFrame(all_rows, columns=["bug", "sequence", "pred_mic", "prob_active", "novelty"])\
      .to_csv(all_path, index=False)
    print(f"[INFO] Top todos guardado: {all_path}")
    
    # Tabla resumen
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    pd.DataFrame(all_results).T.to_csv(summary_csv)
    print(f"[INFO] Métricas guardado: {summary_csv}")
    
    print("\n" + "=" * 70)
    print("✅ GENERACIÓN COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
