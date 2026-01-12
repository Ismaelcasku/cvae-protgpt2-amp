#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 1D — Generación de péptidos con CVAE fine-tuned en bacteria MIC

Mejoras/correcciones clave:
- Generación eficiente con KV-cache (prefijo sólo en el primer forward)
- predict_mic vectorizado (batch) si heads disponibles
- Control robusto de bacteria/condiciones y deduplicación
- Nucleus/top-k sampling correcto + early stop por EOS
- Novelty 3-mers con pre-cómputo (sigue siendo O(N*R), pero más rápido)

Uso típico recomendado: perturbación alrededor de una seed (más estable que prior al inicio).
"""

import os
import json
import argparse
import random
import tempfile
import shutil
import inspect
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel


# ============================================================
# Safe LoRA loading (compatible con versiones antiguas de PEFT)
# ============================================================
def load_lora_adapter_safe(base_model, lora_path: str):
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    sig = inspect.signature(LoraConfig.__init__)
    allowed_init = set(sig.parameters.keys()) - {"self"}
    allowed_meta = {"peft_type", "auto_mapping", "base_model_name_or_path", "revision", "inference_mode"}

    clean_cfg, filtered = {}, []
    for k, v in cfg.items():
        if (k in allowed_init) or (k in allowed_meta):
            clean_cfg[k] = v
        else:
            filtered.append(k)

    if filtered:
        print(f"[INFO] Filtered LoRA config keys: {sorted(filtered)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in os.listdir(lora_path):
            src = os.path.join(lora_path, fname)
            dst = os.path.join(tmpdir, fname)
            if fname != "adapter_config.json" and os.path.isfile(src):
                shutil.copy2(src, dst)
        with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
            json.dump(clean_cfg, f, indent=2)
        model = PeftModel.from_pretrained(base_model, tmpdir)
    return model


# ============================================================
# Encoder AA (debe coincidir con el training)
# ============================================================
AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, MASK, CLS, EOS = 0, 1, 2, 3
AA2ID = {a: i + 4 for i, a in enumerate(AA)}
VOCAB_ENC = 4 + len(AA)


def enc_tokenize_batch(seqs: List[str], max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    ids_list, attn_list = [], []
    for seq in seqs:
        s = str(seq).strip().upper()
        ids = [CLS] + [AA2ID[c] for c in s if c in AA2ID] + [EOS]
        ids = ids[:max_len]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids += [PAD] * pad_n
            attn += [0] * pad_n
        ids_list.append(ids)
        attn_list.append(attn)
    return torch.tensor(ids_list, dtype=torch.long), torch.tensor(attn_list, dtype=torch.bool)


class EncoderAA(nn.Module):
    def __init__(self, latent_dim: int = 64, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(VOCAB_ENC, d_model, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_lv = nn.Linear(d_model, latent_dim)

    def forward(self, ids, attn):
        x = self.emb(ids)
        h = self.enc(x, src_key_padding_mask=~attn.bool())
        h0 = self.norm(h[:, 0])
        mu = self.fc_mu(h0)
        lv = self.fc_lv(h0).clamp(-5, 5)
        return mu, lv


# ============================================================
# Tokens de condicionamiento
# ============================================================
BACTOK = {
    "ecoli": "<ECOLI>",
    "kpneumoniae": "<KPN>",
    "paeruginosa": "<PAER>",
}


# ============================================================
# Sampling helpers
# ============================================================
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    # logits: [B, V]
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        to_remove = cumprobs > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        # scatter back
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, to_remove)
        logits = torch.where(indices_to_remove, torch.full_like(logits, float("-inf")), logits)

    return logits


# ============================================================
# Generador
# ============================================================
class CVAEGenerator:
    def __init__(self, artifacts_dir: str, device: str = "cuda"):
        self.device = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        self.artifacts_dir = artifacts_dir

        meta_path = os.path.join(artifacts_dir, "metadata.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        print(f"[INFO] Loading model from: {artifacts_dir}")
        print(f"[INFO] Mode: {self.meta.get('mode')}")
        print(f"[INFO] Conditions: {self.meta.get('conditions')}")
        print(f"[INFO] Use heads: {self.meta.get('use_heads', False)}")

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(os.path.join(artifacts_dir, "tokenizer"))
        if self.tok.bos_token is None or self.tok.eos_token is None or self.tok.pad_token is None:
            raise ValueError("Tokenizer must have bos/eos/pad tokens saved in artifacts/tokenizer.")
        self.bos = self.tok.bos_token
        self.eos = self.tok.eos_token
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id

        # Decoder + LoRA
        dec_base = AutoModelForCausalLM.from_pretrained(
            "nferruz/ProtGPT2", use_safetensors=True, trust_remote_code=False
        )
        dec_base.resize_token_embeddings(len(self.tok))
        self.hid = dec_base.config.n_embd

        lora_path = os.path.join(artifacts_dir, "lora_adapter")
        self.dec = load_lora_adapter_safe(dec_base, lora_path)
        self.dec.to(self.device).eval()
        print("[INFO] Loaded decoder + LoRA")

        # Encoder
        enc_ckpt = torch.load(os.path.join(artifacts_dir, "encoder.pt"), map_location="cpu")
        self.latent_dim = int(enc_ckpt["hparams"]["latent_dim"])
        self.encoder = EncoderAA(latent_dim=self.latent_dim)
        self.encoder.load_state_dict(enc_ckpt["state_dict"])
        self.encoder.to(self.device).eval()
        print(f"[INFO] Loaded encoder (latent_dim={self.latent_dim})")

        # Prefix MLP
        prefix_ckpt = torch.load(os.path.join(artifacts_dir, "prefix_mlp.pt"), map_location="cpu")
        self.prefix_len = int(prefix_ckpt["hparams"]["prefix_len"])
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.LayerNorm(self.latent_dim * 4),
            nn.GELU(),
            nn.Linear(self.latent_dim * 4, self.prefix_len * self.hid),
        )
        self.prefix_mlp.load_state_dict(prefix_ckpt["state_dict"])
        self.prefix_mlp.to(self.device).eval()
        print(f"[INFO] Loaded prefix_mlp (prefix_len={self.prefix_len})")

        # Heads (optional)
        self.use_heads = bool(self.meta.get("use_heads", False))
        if self.use_heads:
            heads_path = os.path.join(artifacts_dir, "heads.pt")
            if os.path.exists(heads_path):
                heads_ckpt = torch.load(heads_path, map_location="cpu")
                self.reg_head = nn.Linear(self.hid, 1)
                self.cls_head = nn.Linear(self.hid, 1)
                self.reg_head.load_state_dict(heads_ckpt["reg_head"])
                self.cls_head.load_state_dict(heads_ckpt["cls_head"])
                self.reg_head.to(self.device).eval()
                self.cls_head.to(self.device).eval()
                print("[INFO] Loaded MIC prediction heads")
            else:
                print("[WARN] heads.pt not found -> disabling heads")
                self.use_heads = False

        # Bacteria available (seguro)
        conds = self.meta.get("conditions", list(BACTOK.keys()))
        self.bacteria = [c for c in conds if c in BACTOK]
        if not self.bacteria:
            self.bacteria = list(BACTOK.keys())
        print(f"[INFO] Available bacteria: {self.bacteria}")

    @torch.no_grad()
    def encode_mu(self, seqs: List[str], max_len: int = 64) -> torch.Tensor:
        ids, attn = enc_tokenize_batch(seqs, max_len=max_len)
        ids, attn = ids.to(self.device), attn.to(self.device)
        mu, _ = self.encoder(ids, attn)
        return mu

    def sample_z(self, n: int = 1, sigma: float = 1.0) -> torch.Tensor:
        return torch.randn(n, self.latent_dim, device=self.device) * float(sigma)

    def perturb_z(self, mu: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        return mu + torch.randn_like(mu) * float(sigma)

    def interpolate_z(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 5) -> torch.Tensor:
        alphas = torch.linspace(0, 1, steps, device=self.device)
        return torch.stack([z1 * (1 - a) + z2 * a for a in alphas], dim=0)

    @torch.no_grad()
    def generate_batch(
        self,
        z: torch.Tensor,
        bacteria: str,
        max_new_tokens: int = 60,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.95,
    ) -> List[str]:
        bacteria = str(bacteria).strip().lower()
        if bacteria not in BACTOK:
            raise ValueError(f"Unknown bacteria='{bacteria}'. Valid: {list(BACTOK.keys())}")

        B = z.size(0)

        # Prefix
        prefix_flat = self.prefix_mlp(z)
        prefix_emb = prefix_flat.view(B, self.prefix_len, self.hid)

        # Prompt ids: <BAC><BOS>
        prompt = f"{BACTOK[bacteria]}{self.bos}"
        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
        prompt_ids_t = torch.tensor([prompt_ids] * B, device=self.device, dtype=torch.long)

        # Initial forward with inputs_embeds to inject prefix, and enable cache
        prompt_emb = self.dec.get_input_embeddings()(prompt_ids_t)  # [B, P, H]
        inputs_embeds0 = torch.cat([prefix_emb, prompt_emb], dim=1)  # [B, prefix+P, H]
        attn_mask = torch.ones(B, inputs_embeds0.size(1), device=self.device, dtype=torch.long)

        out0 = self.dec(
            inputs_embeds=inputs_embeds0,
            attention_mask=attn_mask,
            use_cache=True,
        )
        past = out0.past_key_values

        # Start generated ids with prompt (sin prefix, porque no son tokens)
        generated = prompt_ids_t.clone()  # [B, P]
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Next token sampling loop (KV-cache)
        next_logits = out0.logits[:, -1, :]  # last position
        for _ in range(int(max_new_tokens)):
            logits = next_logits / max(1e-8, float(temperature))
            logits = top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B,1]

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == self.eos_id)

            # Update attention mask (prefix+prompt+generated_tokens)
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, device=self.device, dtype=torch.long)], dim=1)

            if finished.all():
                break

            out = self.dec(
                input_ids=next_token,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

        # Decode -> keep only AA chars
        seqs = []
        for i in range(B):
            text = self.tok.decode(generated[i].tolist(), skip_special_tokens=True)
            clean = "".join(c for c in text.upper() if c in AA)
            seqs.append(clean)
        return seqs

    @torch.no_grad()
    def predict_mic_batch(self, sequences: List[str], bacteria: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if not self.use_heads:
            return None, None

        bacteria = str(bacteria).strip().lower()
        if bacteria not in BACTOK:
            raise ValueError(f"Unknown bacteria='{bacteria}'. Valid: {list(BACTOK.keys())}")

        B = len(sequences)
        if B == 0:
            return [], []

        # Encode mu (determinista)
        mu = self.encode_mu(sequences)  # [B, D]
        z = mu

        # Prefix
        prefix_flat = self.prefix_mlp(z)
        prefix_emb = prefix_flat.view(B, self.prefix_len, self.hid)

        # Tokenize full text for decoder: <BAC><BOS>SEQ<EOS>, pad batch
        bac_tok = BACTOK[bacteria]
        texts = [f"{bac_tok}{self.bos}{s}{self.eos}" for s in sequences]
        ids_list = [self.tok.encode(t, add_special_tokens=False) for t in texts]
        maxL = max(len(x) for x in ids_list)
        ids_pad = [x + [self.pad_id] * (maxL - len(x)) for x in ids_list]
        attn_tok = [[1] * len(x) + [0] * (maxL - len(x)) for x in ids_list]

        input_ids = torch.tensor(ids_pad, device=self.device, dtype=torch.long)      # [B, L]
        attn = torch.tensor(attn_tok, device=self.device, dtype=torch.long)          # [B, L]
        token_emb = self.dec.get_input_embeddings()(input_ids)                       # [B, L, H]
        inputs_embeds = torch.cat([prefix_emb, token_emb], dim=1)                    # [B, prefix+L, H]
        attn_ext = torch.cat([torch.ones(B, self.prefix_len, device=self.device, dtype=torch.long), attn], dim=1)

        out = self.dec(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_ext,
            output_hidden_states=True,
            use_cache=False,
        )
        last = out.hidden_states[-1]  # [B, prefix+L, H]

        # índice del último token real (en la parte de tokens, no prefix)
        lengths = attn.sum(dim=1).clamp(min=1)  # [B]
        idx = (self.prefix_len + lengths - 1).to(torch.long)  # [B]
        h_T = last[torch.arange(B, device=self.device), idx]   # [B, H]

        mic_pred = self.reg_head(h_T).squeeze(-1)              # [B]
        active_prob = torch.sigmoid(self.cls_head(h_T).squeeze(-1))  # [B]

        return mic_pred.detach().cpu().tolist(), active_prob.detach().cpu().tolist()


# ============================================================
# Novelty (3-mers) con pre-cómputo
# ============================================================
def make_3mer_set(seq: str) -> set:
    s = str(seq).strip().upper()
    if len(s) < 3:
        return set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def compute_novelty_3mer(sequences: List[str], ref_sets: List[set]) -> List[float]:
    if not ref_sets:
        return [1.0] * len(sequences)

    out = []
    for seq in sequences:
        sset = make_3mer_set(seq)
        if not sset:
            out.append(1.0)
            continue
        max_sim = 0.0
        for rset in ref_sets:
            if not rset:
                continue
            inter = len(sset & rset)
            if inter == 0:
                continue
            union = len(sset | rset)
            sim = inter / union if union else 0.0
            if sim > max_sim:
                max_sim = sim
                if max_sim >= 0.999:
                    break
        out.append(1.0 - max_sim)
    return out


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Generate peptides with bacteria-finetuned CVAE")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--n", type=int, default=200, help="Number of UNIQUE sequences per bacteria")
    p.add_argument("--bacteria", type=str, default="ecoli", help="ecoli|kpneumoniae|paeruginosa|all")

    p.add_argument("--mode", choices=["prior", "perturb", "interpolate"], default="perturb")
    p.add_argument("--seed_seq", type=str, default=None)
    p.add_argument("--seed_seq2", type=str, default=None)

    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=60)

    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--max_len", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_tries_factor", type=int, default=40, help="anti-infinite-loop; tries = n * factor")

    p.add_argument("--reference_csv", type=str, default=None, help="CSV with column 'sequence'")
    p.add_argument("--reference_max", type=int, default=0, help="0=no limit; else cap reference size")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gen = CVAEGenerator(args.artifacts_dir, device=args.device)

    # Bacteria list
    if args.bacteria.strip().lower() == "all":
        bacteria_list = gen.bacteria
    else:
        b = args.bacteria.strip().lower()
        if b not in BACTOK:
            raise ValueError(f"--bacteria must be one of {list(BACTOK.keys())} or 'all'")
        bacteria_list = [b]

    # Reference sets for novelty
    ref_sets = []
    if args.reference_csv and os.path.exists(args.reference_csv):
        ref_df = pd.read_csv(args.reference_csv)
        ref_df.columns = [c.strip().lower() for c in ref_df.columns]
        if "sequence" in ref_df.columns:
            ref_seqs = ref_df["sequence"].astype(str).str.upper().str.strip().tolist()
            if args.reference_max and args.reference_max > 0:
                ref_seqs = ref_seqs[: int(args.reference_max)]
            ref_sets = [make_3mer_set(s) for s in ref_seqs]
            print(f"[INFO] Loaded reference sequences: {len(ref_sets)}")

    all_rows = []

    for bac in bacteria_list:
        print(f"\n[INFO] Generating for {bac} | mode={args.mode}")
        target_n = int(args.n)
        max_tries = target_n * int(args.max_tries_factor)
        unique = []
        seen = set()

        tries = 0
        while len(unique) < target_n and tries < max_tries:
            tries += 1

            batch_n = min(int(args.batch_size), target_n - len(unique))

            if args.mode == "prior":
                z = gen.sample_z(batch_n, sigma=args.sigma)

            elif args.mode == "perturb":
                if not args.seed_seq:
                    raise ValueError("--seed_seq is required for mode=perturb")
                mu = gen.encode_mu([args.seed_seq])[0:1, :]
                z = gen.perturb_z(mu.repeat(batch_n, 1), sigma=args.sigma)

            else:  # interpolate
                if not args.seed_seq or not args.seed_seq2:
                    raise ValueError("--seed_seq and --seed_seq2 are required for mode=interpolate")
                mu1 = gen.encode_mu([args.seed_seq])[0:1, :]
                mu2 = gen.encode_mu([args.seed_seq2])[0:1, :]
                z = gen.interpolate_z(mu1, mu2, steps=batch_n).squeeze(1)

            cand = gen.generate_batch(
                z, bac,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            for s in cand:
                s = s.strip().upper()
                if not (args.min_len <= len(s) <= args.max_len):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                unique.append(s)
                if len(unique) >= target_n:
                    break

        if len(unique) < target_n:
            print(f"[WARN] Only generated {len(unique)}/{target_n} unique sequences for {bac} (tries={tries}).")

        # Novelty
        nov = compute_novelty_3mer(unique, ref_sets) if ref_sets else [1.0] * len(unique)

        # Heads prediction (batch)
        mics, actives = gen.predict_mic_batch(unique, bac)

        for i, s in enumerate(unique):
            row = {
                "sequence": s,
                "bacteria": bac,
                "length": len(s),
                "novelty_3mer": float(nov[i]),
            }
            if mics is not None:
                row["predicted_mic"] = float(mics[i])
                row["predicted_active_prob"] = float(actives[i])
            all_rows.append(row)

        if mics is not None:
            active_rate = float(np.mean([a > 0.5 for a in actives])) if actives else 0.0
            print(f"[INFO] {bac}: generated={len(unique)} | active_prob>0.5: {100*active_rate:.1f}%")
        else:
            print(f"[INFO] {bac}: generated={len(unique)}")

    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\n[OK] Saved: {args.output} | rows={len(out_df)} | unique_total={out_df['sequence'].nunique()}")


if __name__ == "__main__":
    main()
