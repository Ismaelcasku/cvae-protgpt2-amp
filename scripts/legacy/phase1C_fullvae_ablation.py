#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 1C — CVAE end-to-end (Full VAE) + Fine-tuning bacteria MIC

v3: 
  - Added delta_lora_off ablation to separate LoRA vs Prefix effects
  - Improved ablation diagnostics
  - All previous features from v2
"""

import os
import math
import json
import inspect
import argparse
import random
import tempfile
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


# ============================================================
# Reproducibility
# ============================================================
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Safe LoRA adapter loading (compatible with older PEFT versions)
# ============================================================
def load_lora_adapter_safe(base_model, lora_path: str):
    """
    Carga un LoRA adapter aunque adapter_config.json tenga campos de PEFT más nuevo.
    """
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    sig = inspect.signature(LoraConfig.__init__)
    allowed_init = set(sig.parameters.keys()) - {"self"}
    
    allowed_meta = {
        "peft_type",
        "auto_mapping",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
    }
    
    clean_cfg = {}
    filtered = []
    for k, v in cfg.items():
        if (k in allowed_init) or (k in allowed_meta):
            clean_cfg[k] = v
        else:
            filtered.append(k)
    
    if filtered:
        print(f"[INFO] Filtered LoRA config keys not supported by this PEFT: {sorted(filtered)}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in os.listdir(lora_path):
            src = os.path.join(lora_path, fname)
            dst = os.path.join(tmpdir, fname)
            if fname == "adapter_config.json":
                continue
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        
        with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
            json.dump(clean_cfg, f, indent=2)
        
        model = PeftModel.from_pretrained(base_model, tmpdir)
    
    return model


# ============================================================
# Conditioning tokens
# ============================================================
BACTOK = {
    "ecoli": "<ECOLI>",
    "kpneumoniae": "<KPN>",
    "paeruginosa": "<PAER>",
}

ACTTOK = {
    "amp": "<AMP>",
    "noamp": "<NOAMP>",
}

CONDTOK = {**BACTOK, **ACTTOK}


# ============================================================
# Encoder AA vocab
# ============================================================
AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, MASK, CLS, EOS = 0, 1, 2, 3
AA2ID = {a: i + 4 for i, a in enumerate(AA)}
VOCAB_ENC = 4 + len(AA)


def enc_tokenize(seq: str, max_len: int):
    seq = str(seq).strip().upper()
    ids = [CLS] + [AA2ID[c] for c in seq if c in AA2ID] + [EOS]
    ids = ids[:max_len]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        padn = max_len - len(ids)
        ids += [PAD] * padn
        attn += [0] * padn
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)


# ============================================================
# Dataset loader
# ============================================================
class LongCSV(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "sequence" not in df.columns:
            raise ValueError(f"CSV must have 'sequence' column. Found: {list(df.columns)}")

        self.bugs = [b for b in BACTOK if b in df.columns]
        has_activity = ("activity" in df.columns) or ("label" in df.columns)

        if self.bugs:
            self.mode = "bacteria"
            self.conditions = self.bugs
            print(f"[INFO] Dataset mode: BACTERIA MIC (columns: {self.bugs})")

            rows = []
            for _, r in df.iterrows():
                seq = str(r["sequence"]).strip().upper()
                for b in self.bugs:
                    v = r[b]
                    if pd.isna(v) or (isinstance(v, str) and v.strip().upper() in ("NA", "N/A", "")):
                        rows.append((seq, b, None))
                    else:
                        try:
                            rows.append((seq, b, float(v)))
                        except Exception:
                            rows.append((seq, b, None))
            self.data = rows

        elif has_activity:
            self.mode = "activity"
            self.conditions = list(ACTTOK.keys())
            print("[INFO] Dataset mode: ACTIVITY CLASS (amp/noamp)")

            rows = []
            for _, r in df.iterrows():
                seq = str(r["sequence"]).strip().upper()

                if "activity" in df.columns:
                    act = str(r["activity"]).strip().lower()
                    cond = "amp" if act in ("amp", "1", "true", "yes") else "noamp"
                else:
                    lab = r["label"]
                    cond = "amp" if (not pd.isna(lab) and int(lab) == 1) else "noamp"

                rows.append((seq, cond, None))
            self.data = rows

        else:
            raise ValueError(
                f"CSV format not recognized. Found columns: {list(df.columns)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


class CVAEDataset(Dataset):
    def __init__(self, base_subset, tok, max_len_dec: int, max_len_enc: int, mic_active_thr: float = 4.0):
        self.base = base_subset
        self.tok = tok
        self.max_len_dec = int(max_len_dec)
        self.max_len_enc = int(max_len_enc)
        self.pad_id = tok.pad_token_id
        self.bos = tok.bos_token
        self.eos = tok.eos_token
        self.mic_active_thr = float(mic_active_thr)

        self._cond_lens = {}
        for cond, cond_tok in CONDTOK.items():
            cond_text = f"{cond_tok}{self.bos}"
            cond_ids = tok.encode(cond_text, add_special_tokens=False)
            self._cond_lens[cond] = len(cond_ids)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        seq, cond, mic = self.base[i]
        seq = str(seq).strip().upper()

        cond_token = CONDTOK[cond]
        text = f"{cond_token}{self.bos}{seq}{self.eos}"

        ids = self.tok.encode(text, add_special_tokens=False)
        ids = ids[: self.max_len_dec + 3]

        attn = [1] * len(ids)
        labels = ids.copy()

        n_cond = self._cond_lens[cond]
        for j in range(min(n_cond, len(labels))):
            labels[j] = -100

        while len(ids) < self.max_len_dec + 3:
            ids.append(self.pad_id)
            attn.append(0)
            labels.append(-100)

        bin_y = -1.0 if mic is None else (1.0 if mic < self.mic_active_thr else 0.0)
        reg_y = -1.0 if mic is None else float(mic)

        enc_ids, enc_attn = enc_tokenize(seq, self.max_len_enc)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attn": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "enc_ids": enc_ids,
            "enc_attn": enc_attn,
            "bin_y": torch.tensor(bin_y, dtype=torch.float32),
            "reg_y": torch.tensor(reg_y, dtype=torch.float32),
            "seq": seq,
            "cond": cond,
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        if k in ("seq", "cond"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ============================================================
# Encoder
# ============================================================
class EncoderAA(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)

        self.emb = nn.Embedding(VOCAB_ENC, d_model, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, self.latent_dim)
        self.fc_lv = nn.Linear(d_model, self.latent_dim)

    def forward(self, ids, attn):
        x = self.emb(ids)
        h = self.enc(x, src_key_padding_mask=~attn.bool())
        h0 = self.norm(h[:, 0])
        mu = self.fc_mu(h0)
        lv = self.fc_lv(h0).clamp(-5, 5)
        return mu, lv


def reparameterize(mu, lv):
    std = torch.exp(0.5 * lv)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_per_dim(mu, lv):
    return 0.5 * (lv.exp() + mu.pow(2) - 1.0 - lv)


def load_encoder_pretrained(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {ckpt_path} missing 'state_dict'")

    hparams = ckpt.get("hparams", {})
    latent_dim = int(hparams.get("latent_dim", 64))

    valid_prefixes = ("emb.", "enc.", "norm.", "fc_mu.", "fc_lv.")
    exclude_patterns = ("cls_head", "auroc", "auprc", "acc", "f1", "_metric")

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        kl = k.lower()
        if any(p in kl for p in exclude_patterns):
            continue
        if any(k.startswith(pfx) for pfx in valid_prefixes):
            state_dict[k] = v

    return state_dict, {"latent_dim": latent_dim}


# ============================================================
# LitCVAE
# ============================================================
class LitCVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 64,
        prefix_len: int = 16,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        kl_warmup_frac: float = 0.4,
        free_bits: float = 0.10,
        w_lm: float = 1.0,
        w_kl: float = 1.0,
        w_reg: float = 0.3,
        w_cls: float = 0.5,
        w_prefix_norm: float = 0.01,
        target_prefix_norm: float = 15.0,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.15,
        lora_targets: str = "c_attn,c_proj",
        max_epochs: int = 100,
        use_heads: bool = True,
        mic_active_thr: float = 4.0,
        ablation_every_n_epochs: int = 5,
        encoder_ckpt: str = None,
        freeze_encoder_epochs: int = 0,
        pretrained_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._encoder_frozen = False
        self.mic_active_thr = float(mic_active_thr)
        self.is_finetuning = bool(pretrained_dir and os.path.isdir(pretrained_dir))
        self.new_tokens_added = []

        if self.is_finetuning:
            print("=" * 60)
            print("[INFO] FINE-TUNING mode")
            print(f"[INFO] pretrained_dir: {pretrained_dir}")
            print("=" * 60)

            tok_path = os.path.join(pretrained_dir, "tokenizer")
            self.tok = AutoTokenizer.from_pretrained(tok_path)
            print(f"[INFO] Loaded tokenizer: {len(self.tok)} tokens")

            special_to_add = {}
            if self.tok.bos_token is None:
                special_to_add["bos_token"] = "<BOS>"
            if self.tok.eos_token is None:
                special_to_add["eos_token"] = "<EOS>"
            if self.tok.pad_token is None:
                special_to_add["pad_token"] = "<PAD>"
            if special_to_add:
                self.tok.add_special_tokens(special_to_add)

            existing = set(self.tok.get_vocab().keys())
            missing_cond = [t for t in CONDTOK.values() if t not in existing]
            if missing_cond:
                self.tok.add_special_tokens({"additional_special_tokens": missing_cond})
                self.new_tokens_added = missing_cond
                print(f"[INFO] Added new conditioning tokens: {missing_cond}")

            lora_path = os.path.join(pretrained_dir, "lora_adapter")
            dec_base = AutoModelForCausalLM.from_pretrained(
                "nferruz/ProtGPT2", use_safetensors=True, trust_remote_code=False
            )
            dec_base.resize_token_embeddings(len(self.tok))
            self.hid = dec_base.config.n_embd

            self.dec = load_lora_adapter_safe(dec_base, lora_path)

            for name, p in self.dec.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

            if self.new_tokens_added:
                emb = self.dec.get_input_embeddings()
                emb.weight.requires_grad = True
                out_emb = self.dec.get_output_embeddings()
                if out_emb is not None and hasattr(out_emb, "weight"):
                    out_emb.weight.requires_grad = True
                print("[INFO] New tokens -> enabling embedding training")

            print(f"[INFO] Loaded LoRA adapter: {lora_path}")
            self.dec.print_trainable_parameters()

            enc_path = os.path.join(pretrained_dir, "encoder.pt")
            enc_ckpt = torch.load(enc_path, map_location="cpu")
            self.encoder = EncoderAA(latent_dim=int(latent_dim))
            self.encoder.load_state_dict(enc_ckpt["state_dict"])
            print(f"[INFO] Loaded encoder.pt")

            prefix_path = os.path.join(pretrained_dir, "prefix_mlp.pt")
            prefix_ckpt = torch.load(prefix_path, map_location="cpu")
            prefix_hparams = prefix_ckpt["hparams"]

            self.prefix_len = int(prefix_hparams["prefix_len"])
            self.prefix_mlp = nn.Sequential(
                nn.Linear(int(latent_dim), int(latent_dim) * 4),
                nn.LayerNorm(int(latent_dim) * 4),
                nn.GELU(),
                nn.Linear(int(latent_dim) * 4, self.prefix_len * self.hid),
            )
            self.prefix_mlp.load_state_dict(prefix_ckpt["state_dict"])
            print(f"[INFO] Loaded prefix_mlp.pt (prefix_len={self.prefix_len})")

        else:
            print("=" * 60)
            print("[INFO] TRAINING from scratch")
            print("=" * 60)

            self.tok = AutoTokenizer.from_pretrained("nferruz/ProtGPT2", trust_remote_code=False)

            special_to_add = {}
            if self.tok.bos_token is None:
                special_to_add["bos_token"] = "<BOS>"
            if self.tok.eos_token is None:
                special_to_add["eos_token"] = "<EOS>"
            if self.tok.pad_token is None:
                special_to_add["pad_token"] = "<PAD>"
            if special_to_add:
                self.tok.add_special_tokens(special_to_add)

            self.tok.add_special_tokens({"additional_special_tokens": list(CONDTOK.values())})

            dec_base = AutoModelForCausalLM.from_pretrained(
                "nferruz/ProtGPT2", use_safetensors=True, trust_remote_code=False
            )
            dec_base.resize_token_embeddings(len(self.tok))
            self.hid = dec_base.config.n_embd

            targets = [t.strip() for t in lora_targets.split(",") if t.strip()]
            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.dec = get_peft_model(dec_base, lora_cfg)
            self.dec.print_trainable_parameters()

            self.encoder = EncoderAA(latent_dim=int(latent_dim))
            if encoder_ckpt and os.path.isfile(encoder_ckpt):
                state_dict, _ = load_encoder_pretrained(encoder_ckpt)
                self.encoder.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded encoder ckpt: {encoder_ckpt}")

            self.prefix_len = int(prefix_len)
            self.prefix_mlp = nn.Sequential(
                nn.Linear(int(latent_dim), int(latent_dim) * 4),
                nn.LayerNorm(int(latent_dim) * 4),
                nn.GELU(),
                nn.Linear(int(latent_dim) * 4, self.prefix_len * self.hid),
            )

        for name, tok_str in CONDTOK.items():
            ids = self.tok.encode(tok_str, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Token '{tok_str}' must be single token")

        self.bos_token = self.tok.bos_token
        self.eos_token = self.tok.eos_token
        self.pad_token_id = self.tok.pad_token_id
        self.eos_token_id = self.tok.eos_token_id

        self.use_heads = bool(use_heads)
        if self.use_heads:
            self.reg_head = nn.Linear(self.hid, 1)
            self.cls_head = nn.Linear(self.hid, 1)
            self.huber = nn.HuberLoss()
            self.bce = nn.BCEWithLogitsLoss()
            print("[INFO] MIC heads enabled")

    def on_train_epoch_start(self):
        freeze_epochs = int(self.hparams.freeze_encoder_epochs)
        if freeze_epochs <= 0:
            return
        if self.current_epoch < freeze_epochs and not self._encoder_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self._encoder_frozen = True
        if self.current_epoch >= freeze_epochs and self._encoder_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = True
            self._encoder_frozen = False

    def beta(self) -> float:
        warm = max(1, int(self.trainer.max_epochs * float(self.hparams.kl_warmup_frac)))
        t = min(1.0, float(self.current_epoch) / float(warm))
        return float(self.hparams.beta_min) + t * (float(self.hparams.beta_max) - float(self.hparams.beta_min))

    def forward_decoder_with_prefix(self, input_ids, attn, labels, prefix_emb):
        B = input_ids.size(0)
        device = input_ids.device

        token_emb = self.dec.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_emb, token_emb], dim=1)

        prefix_attn = torch.ones(B, self.prefix_len, device=device, dtype=attn.dtype)
        prefix_labels = torch.full((B, self.prefix_len), -100, device=device, dtype=labels.dtype)

        attn_ext = torch.cat([prefix_attn, attn], dim=1)
        labels_ext = torch.cat([prefix_labels, labels], dim=1)

        return self.dec(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_ext,
            labels=labels_ext,
            output_hidden_states=True,
            use_cache=False,
        )

    def compute_kl(self, mu, lv):
        kl = kl_per_dim(mu, lv)
        fb = float(self.hparams.free_bits)
        kl_fb = torch.clamp(kl, min=fb).sum(dim=-1).mean()
        kl_total = kl.sum(dim=-1).mean()
        with torch.no_grad():
            active_units = (kl.mean(dim=0) > fb).float().sum()
        return kl_fb, kl_total, active_units

    @staticmethod
    def participation_ratio(z):
        with torch.no_grad():
            z_f32 = z.detach().to("cpu", dtype=torch.float32)
            if z_f32.size(0) < 8:
                return torch.tensor(float("nan"))
            try:
                cov = torch.cov(z_f32.T)
                cov = cov + 1e-6 * torch.eye(cov.size(0), dtype=cov.dtype)
                eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-10)
                return (eigs.sum() ** 2) / (eigs ** 2).sum()
            except Exception:
                return torch.tensor(float("nan"))

    def compute_ablation(self, batch, z, lm_loss_real):
        """
        Ablation completa: shuffle, zero, prefix_off, lora_off
        """
        with torch.no_grad():
            B = z.size(0)
            
            # 1. Shuffle ablation
            perm = torch.randperm(B, device=z.device)
            z_shuffled = z[perm]
            prefix_shuffled = self.prefix_mlp(z_shuffled).view(B, self.prefix_len, self.hid)
            out_shuffled = self.forward_decoder_with_prefix(batch["input_ids"], batch["attn"], batch["labels"], prefix_shuffled)
            lm_shuffled = out_shuffled.loss
            delta_shuffle = lm_shuffled - lm_loss_real

            # 2. Zero ablation
            z_zero = torch.zeros_like(z)
            prefix_zero = self.prefix_mlp(z_zero).view(B, self.prefix_len, self.hid)
            out_zero = self.forward_decoder_with_prefix(batch["input_ids"], batch["attn"], batch["labels"], prefix_zero)
            lm_zero = out_zero.loss
            delta_zero = lm_zero - lm_loss_real
            
            # 3. Prefix OFF ablation (LoRA ON, no prefix)
            out_no_prefix = self.dec(
                input_ids=batch["input_ids"],
                attention_mask=batch["attn"],
                labels=batch["labels"],
                use_cache=False,
            )
            lm_no_prefix = out_no_prefix.loss
            delta_prefix_off = lm_no_prefix - lm_loss_real
            
            # 4. LoRA OFF ablation (LoRA OFF, prefix ON) - NUEVO
            # Solo si el modelo tiene LoRA adapters
            has_lora = any("lora_" in n for n, _ in self.dec.named_parameters())
            
            if has_lora:
                try:
                    self.dec.disable_adapter_layers()
                    prefix_real = self.prefix_mlp(z).view(B, self.prefix_len, self.hid)
                    out_no_lora = self.forward_decoder_with_prefix(batch["input_ids"], batch["attn"], batch["labels"], prefix_real)
                    lm_no_lora = out_no_lora.loss
                    delta_lora_off = lm_no_lora - lm_loss_real
                    self.dec.enable_adapter_layers()
                except Exception as e:
                    print(f"[WARN] LoRA ablation failed: {e}")
                    delta_lora_off = torch.tensor(float('nan'), device=z.device)
            else:
                delta_lora_off = torch.tensor(float('nan'), device=z.device)

            return delta_shuffle, delta_zero, lm_shuffled, lm_zero, delta_prefix_off, lm_no_prefix, delta_lora_off, lm_no_lora if has_lora else torch.tensor(float('nan'), device=z.device)

    def on_validation_epoch_start(self):
        self._ablation_done_this_epoch = False

    def _step(self, batch, stage: str):
        mu, lv = self.encoder(batch["enc_ids"], batch["enc_attn"])
        z = reparameterize(mu, lv)

        prefix_flat = self.prefix_mlp(z)
        prefix_emb = prefix_flat.view(-1, self.prefix_len, self.hid)

        out = self.forward_decoder_with_prefix(batch["input_ids"], batch["attn"], batch["labels"], prefix_emb)
        lm_loss = out.loss

        kl_fb, kl_total, active_units = self.compute_kl(mu, lv)
        beta = self.beta()
        kl_term = beta * kl_fb

        prefix_norm = prefix_emb.norm(dim=-1).mean()
        prefix_pen = (prefix_norm - float(self.hparams.target_prefix_norm)).abs()

        reg_loss = lm_loss * 0.0
        cls_loss = lm_loss * 0.0

        if self.use_heads:
            last = out.hidden_states[-1]
            idx = batch["attn"].sum(dim=1).clamp(min=1) - 1 + self.prefix_len
            h_T = last[torch.arange(last.size(0), device=last.device), idx]

            reg_pred = self.reg_head(h_T).squeeze(-1)
            cls_log = self.cls_head(h_T).squeeze(-1)

            reg_mask = batch["reg_y"] >= 0
            cls_mask = batch["bin_y"] >= 0

            if reg_mask.any():
                reg_loss = self.huber(reg_pred[reg_mask], batch["reg_y"][reg_mask])
            if cls_mask.any():
                cls_loss = self.bce(cls_log[cls_mask], batch["bin_y"][cls_mask])

        loss = (
            float(self.hparams.w_lm) * lm_loss
            + float(self.hparams.w_kl) * kl_term
            + float(self.hparams.w_prefix_norm) * prefix_pen
            + float(self.hparams.w_reg) * reg_loss
            + float(self.hparams.w_cls) * cls_loss
        )

        with torch.no_grad():
            z_std = mu.std(dim=0).mean()
            z_norm = z.norm(dim=-1).mean()

        logs = {
            f"{stage}_loss": loss,
            f"{stage}_lm": lm_loss,
            f"{stage}_kl_fb": kl_fb,
            f"{stage}_kl_total": kl_total,
            f"{stage}_active_units": active_units,
            f"{stage}_beta": torch.tensor(beta, device=self.device),
            f"{stage}_prefix_norm": prefix_norm,
            f"{stage}_z_std": z_std,
            f"{stage}_z_norm": z_norm,
        }

        if self.use_heads:
            logs[f"{stage}_reg"] = reg_loss
            logs[f"{stage}_cls"] = cls_loss

        if stage == "val":
            logs[f"{stage}_z_effdim"] = self.participation_ratio(mu)
            ablation_every = int(self.hparams.ablation_every_n_epochs)
            if ablation_every > 0 and (self.current_epoch % ablation_every == 0):
                if not getattr(self, "_ablation_done_this_epoch", False):
                    dsh, dz0, lm_shuf, lm0, dpoff, lm_nopfx, dloff, lm_nolora = self.compute_ablation(batch, z, lm_loss)
                    logs[f"{stage}_lm_shuffled"] = lm_shuf
                    logs[f"{stage}_lm_zero"] = lm0
                    logs[f"{stage}_lm_no_prefix"] = lm_nopfx
                    logs[f"{stage}_lm_no_lora"] = lm_nolora
                    logs[f"{stage}_delta_shuffle"] = dsh
                    logs[f"{stage}_delta_zero"] = dz0
                    logs[f"{stage}_delta_prefix_off"] = dpoff
                    logs[f"{stage}_delta_lora_off"] = dloff  # NUEVO
                    self._ablation_done_this_epoch = True

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(self.hparams.max_epochs), eta_min=float(self.hparams.lr) * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len_dec", type=int, default=64)
    ap.add_argument("--max_len_enc", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--encoder_ckpt", type=str, default=None)
    ap.add_argument("--freeze_encoder_epochs", type=int, default=0)
    ap.add_argument("--pretrained_dir", type=str, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--beta_min", type=float, default=0.0)
    ap.add_argument("--beta_max", type=float, default=1.0)
    ap.add_argument("--kl_warmup_frac", type=float, default=0.4)
    ap.add_argument("--free_bits", type=float, default=0.10)
    ap.add_argument("--w_lm", type=float, default=1.0)
    ap.add_argument("--w_kl", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.3)
    ap.add_argument("--w_cls", type=float, default=0.5)
    ap.add_argument("--use_heads", action="store_true")
    ap.add_argument("--mic_active_thr", type=float, default=4.0)
    ap.add_argument("--w_prefix_norm", type=float, default=0.01)
    ap.add_argument("--target_prefix_norm", type=float, default=15.0)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.15)
    ap.add_argument("--lora_targets", type=str, default="c_attn,c_proj")
    ap.add_argument("--ablation_every_n_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=12)

    args = ap.parse_args()

    seed_all(42)
    torch.set_float32_matmul_precision("high")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    art_dir = os.path.join(args.out_dir, "artifacts")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 1C CVAE (v3: improved ablations)")
    print("=" * 60)

    base = LongCSV(args.csv)
    print(f"[INFO] Dataset: {len(base)} rows, mode={base.mode}")

    use_heads_effective = bool(args.use_heads)
    if use_heads_effective and base.mode == "activity":
        print("[INFO] Disabling heads for activity mode")
        use_heads_effective = False

    dataset_mode = base.mode
    dataset_conditions = base.conditions

    seq2idx = {}
    for i, (seq, cond, mic) in enumerate(base.data):
        seq2idx.setdefault(seq, []).append(i)

    all_seqs = list(seq2idx.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_seqs)

    n_tr = int(0.9 * len(all_seqs))
    train_seqs = set(all_seqs[:n_tr])
    val_seqs = set(all_seqs[n_tr:])

    train_idx, val_idx = [], []
    for seq, idxs in seq2idx.items():
        (train_idx if seq in train_seqs else val_idx).extend(idxs)

    tr_subset = torch.utils.data.Subset(base, train_idx)
    va_subset = torch.utils.data.Subset(base, val_idx)

    print(f"[INFO] Split: train={len(tr_subset)}, val={len(va_subset)}")

    model = LitCVAE(
        latent_dim=args.latent_dim,
        prefix_len=args.prefix_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        kl_warmup_frac=args.kl_warmup_frac,
        free_bits=args.free_bits,
        w_lm=args.w_lm,
        w_kl=args.w_kl,
        w_reg=args.w_reg,
        w_cls=args.w_cls,
        w_prefix_norm=args.w_prefix_norm,
        target_prefix_norm=args.target_prefix_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        max_epochs=args.epochs,
        use_heads=use_heads_effective,
        mic_active_thr=args.mic_active_thr,
        ablation_every_n_epochs=args.ablation_every_n_epochs,
        encoder_ckpt=args.encoder_ckpt,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        pretrained_dir=args.pretrained_dir,
    )

    dl_tr = DataLoader(
        CVAEDataset(tr_subset, model.tok, args.max_len_dec, args.max_len_enc, args.mic_active_thr),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )
    dl_va = DataLoader(
        CVAEDataset(va_subset, model.tok, args.max_len_dec, args.max_len_enc, args.mic_active_thr),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )

    logger = CSVLogger(save_dir=os.path.join(args.out_dir, "logs"), name="phase1C_cvae")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch:03d}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )

    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_ckpt = resume_ckpt if os.path.isfile(resume_ckpt) else None

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=[
            ckpt_cb,
            EarlyStopping("val_loss", patience=args.patience, mode="min"),
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dl_tr, dl_va, ckpt_path=resume_ckpt)

    best_ckpt_path = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    best_score_f = float(best_score) if best_score is not None else None
    
    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        print(f"\n[INFO] Loading best checkpoint for artifacts: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        
        if "state_dict" in best_ckpt:
            missing, unexpected = model.load_state_dict(best_ckpt["state_dict"], strict=False)
            if missing or unexpected:
                print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if best_score_f is not None:
                print(f"[INFO] Loaded best model state (val_loss={best_score_f:.4f})")
        else:
            print("[WARN] best checkpoint has no 'state_dict' key, using current model state")
    else:
        print("[WARN] No best checkpoint found, saving current model state")

    print("\n[INFO] Saving artifacts from BEST checkpoint...")

    model.tok.save_pretrained(os.path.join(art_dir, "tokenizer"))

    save_embed = bool(getattr(model, "new_tokens_added", []))
    try:
        model.dec.save_pretrained(os.path.join(art_dir, "lora_adapter"), save_embedding_layers=save_embed)
    except TypeError:
        print("[WARN] PEFT does not support save_embedding_layers; saving adapter without it.")
        model.dec.save_pretrained(os.path.join(art_dir, "lora_adapter"))

    torch.save(
        {"state_dict": model.encoder.state_dict(), "hparams": {"latent_dim": args.latent_dim}},
        os.path.join(art_dir, "encoder.pt"),
    )

    torch.save(
        {
            "state_dict": model.prefix_mlp.state_dict(),
            "hparams": {"latent_dim": args.latent_dim, "prefix_len": model.prefix_len, "hid": model.hid},
        },
        os.path.join(art_dir, "prefix_mlp.pt"),
    )

    if use_heads_effective:
        torch.save(
            {"reg_head": model.reg_head.state_dict(), "cls_head": model.cls_head.state_dict()},
            os.path.join(art_dir, "heads.pt"),
        )

    torch.save({"state_dict": model.state_dict(), "hparams": dict(model.hparams)}, os.path.join(art_dir, "model_full.pt"))

    metadata = {
        "mode": dataset_mode,
        "conditions": dataset_conditions,
        "tokens": {c: CONDTOK[c] for c in dataset_conditions},
        "use_heads": use_heads_effective,
        "latent_dim": args.latent_dim,
        "prefix_len": model.prefix_len,
        "hid": model.hid,
        "mic_active_thr": args.mic_active_thr,
        "finetuning": bool(args.pretrained_dir),
        "new_tokens_added": list(getattr(model, "new_tokens_added", [])),
        "checkpoint_monitor": "val_loss",
        "best_checkpoint": best_ckpt_path,
        "best_val_loss": best_score_f,
        "loss_weights": {
            "w_lm": float(args.w_lm),
            "w_kl": float(args.w_kl),
            "w_reg": float(args.w_reg),
            "w_cls": float(args.w_cls),
            "w_prefix_norm": float(args.w_prefix_norm),
        },
        "kl_config": {
            "beta_min": float(args.beta_min),
            "beta_max": float(args.beta_max),
            "kl_warmup_frac": float(args.kl_warmup_frac),
            "free_bits": float(args.free_bits),
        },
        "ablations_enabled": {
            "delta_shuffle": True,
            "delta_zero": True,
            "delta_prefix_off": True,
            "delta_lora_off": True,  # NUEVO
        }
    }
    with open(os.path.join(art_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"[OK] Best checkpoint: {best_ckpt_path}")
    if best_score_f is not None:
        print(f"[OK] Best val_loss: {best_score_f:.4f}")
    print(f"[OK] Artifacts saved from best model: {art_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
