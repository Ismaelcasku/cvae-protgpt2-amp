#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 1C Latent2 + LoRA (v2 - Anti-overfitting)
- Encoder 1A congelado → z externo
- ProtGPT2 decoder con LoRA
- Bridge W_align entrenable
- Regularización de z_norm para forzar uso del latente
- Split por SECUENCIA (sin leakage)

Mejoras anti-overfitting:
- Early stopping patience reducido (10)
- LoRA dropout aumentado (0.15)
- LoRA rank reducido (4)
- Weight decay en AdamW
- Cosine annealing LR scheduler

Estructura de salida:
    out_dir/
    ├── logs/phase1C_lora/version_X/metrics.csv
    ├── checkpoints/
    │   ├── epoch=xxx-val_lm=x.xxx.ckpt
    │   └── last.ckpt
    └── artifacts/
        ├── model_full.pt
        ├── lora_adapter/
        └── tokenizer/
"""

import os
import sys
import argparse
import random
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
from peft import LoraConfig, get_peft_model


# ============================================================
# Utils
# ============================================================

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

BACTOK = {
    "ecoli": "<ECOLI>",
    "kpneumoniae": "<KPN>",
    "paeruginosa": "<PAER>",
}


# ============================================================
# Dataset (idéntico a phase1C_latent2_2.py)
# ============================================================

class LongCSV(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        assert "sequence" in df.columns

        self.bugs = [b for b in BACTOK if b in df.columns]
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
                    except:
                        rows.append((seq, b, None))

        self.data = rows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class LMCondDataset(Dataset):
    def __init__(self, base, tok, max_len):
        self.base = base
        self.tok = tok
        self.max_len = max_len
        self.pad = tok.pad_token_id
        self.bos = tok.bos_token or "<BOS>"
        self.eos = tok.eos_token or "<EOS>"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        seq, bug, y = self.base[i]

        text = f"{BACTOK[bug]}{self.bos}{seq}{self.eos}"
        ids = self.tok.encode(text, add_special_tokens=False)
        ids = ids[: self.max_len + 3]

        attn = [1] * len(ids)
        labels = ids.copy()

        if len(labels) > 0:
            labels[0] = -100
        if len(labels) > 1:
            labels[1] = -100

        while len(ids) < self.max_len + 3:
            ids.append(self.pad)
            attn.append(0)
            labels.append(-100)

        bin_y = -1.0 if y is None else (1.0 if y < 4.0 else 0.0)
        reg_y = -1.0 if y is None else float(y)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attn": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "bin_y": torch.tensor(bin_y, dtype=torch.float32),
            "reg_y": torch.tensor(reg_y, dtype=torch.float32),
            "seq": seq,
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "seq":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ============================================================
# Encoder 1A loader
# ============================================================

def load_encoder_1A(ckpt_path):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    from phase1A_v2 import EncoderVAE

    ckpt = torch.load(ckpt_path, map_location="cpu")
    h = ckpt["hparams"]

    enc = EncoderVAE(latent_dim=int(h["latent_dim"]))
    enc.load_state_dict(ckpt["state_dict"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    return enc, int(h["max_len"]), int(h["latent_dim"])


# ============================================================
# Lightning Module
# ============================================================

class LitFTLatent2LoRA(pl.LightningModule):

    def __init__(
        self,
        encoder_ckpt: str,
        lr: float,
        weight_decay: float,
        zdim: int,
        w_lm: float,
        w_reg: float,
        w_cls: float,
        w_align: float,
        w_znorm: float,
        target_znorm: float,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        max_epochs: int,  # para cosine scheduler
    ):
        super().__init__()
        self.save_hyperparameters()

        # ============================================================
        # Tokenizer + Decoder base
        # ============================================================
        print("[INFO] Cargando ProtGPT2 base...")
        
        self.tok = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.tok.add_special_tokens({"additional_special_tokens": list(BACTOK.values())})
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({"pad_token": "<PAD>"})

        dec_base = AutoModelForCausalLM.from_pretrained(
            "nferruz/ProtGPT2",
            use_safetensors=True
        )
        dec_base.resize_token_embeddings(len(self.tok))

        # Guardar hidden size ANTES de aplicar LoRA
        self.hid = dec_base.config.n_embd

        # ============================================================
        # Aplicar LoRA
        # ============================================================
        print(f"[INFO] Aplicando LoRA (r={lora_r}, dropout={lora_dropout})...")
        
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.dec = get_peft_model(dec_base, lora_cfg)
        self.dec.print_trainable_parameters()

        # ============================================================
        # Heads (sobre h_T)
        # ============================================================
        self.reg_head = nn.Linear(self.hid, 1)
        self.cls_head = nn.Linear(self.hid, 1)

        # ============================================================
        # Encoder 1A (congelado)
        # ============================================================
        enc, enc_max_len, enc_latent_dim = load_encoder_1A(encoder_ckpt)
        
        if enc_latent_dim != zdim:
            raise ValueError(f"zdim={zdim} != encoder latent_dim={enc_latent_dim}")

        self.encoder_1A = enc
        self.enc_max_len = enc_max_len
        self._enc_device = "cpu"

        print(f"[INFO] Encoder 1A cargado: latent_dim={enc_latent_dim}")

        # ============================================================
        # Bridge: z_enc → h_T
        # ============================================================
        self.W_align = nn.Linear(zdim, self.hid)

        # ============================================================
        # Losses
        # ============================================================
        self.huber = nn.HuberLoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------
    def on_fit_start(self):
        device = next(self.dec.parameters()).device
        self.encoder_1A = self.encoder_1A.to(device)
        self._enc_device = str(device)
        print(f"[INFO] Encoder 1A movido a device: {self._enc_device}")

    # ------------------------------------------------------------
    # Metrics helpers (ROBUSTOS)
    # ------------------------------------------------------------
    @staticmethod
    def participation_ratio(z):
        """
        Participation ratio: dimensionalidad efectiva del espacio latente.
        CPU + float32 para estabilidad numérica.
        """
        with torch.no_grad():
            z_f32 = z.detach().to("cpu", dtype=torch.float32)

            if z_f32.size(0) < 8:
                return torch.tensor(float("nan"))

            try:
                cov = torch.cov(z_f32.T)
                eps = 1e-6
                cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
                eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-10)
                pr = (eigs.sum() ** 2) / (eigs ** 2).sum()
                return pr
            except (RuntimeError, torch._C._LinAlgError):
                return torch.tensor(float("nan"))

    # ------------------------------------------------------------
    # Annealing (igual que versión sin LoRA)
    # ------------------------------------------------------------
    def lambda_align(self):
        e = int(self.current_epoch)
        if e < 3:
            return 0.0
        if e < 10:
            return float(self.hparams.w_align) * (e - 2) / 7.0
        return float(self.hparams.w_align)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels):
        return self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )

    # ------------------------------------------------------------
    # Step compartido
    # ------------------------------------------------------------
    def _step(self, batch, stage: str):
        out = self(batch["input_ids"], batch["attn"], batch["labels"])
        lm_loss = out.loss

        # Extraer h_T
        last = out.hidden_states[-1]
        idx = batch["attn"].sum(dim=1).clamp(min=1) - 1
        h_T = last[torch.arange(last.size(0), device=last.device), idx]

        # Heads
        reg_pred = self.reg_head(h_T).squeeze(-1)
        cls_log = self.cls_head(h_T).squeeze(-1)

        # Encoder z
        with torch.no_grad():
            z_enc, _ = self.encoder_1A.encode_sequences(
                batch["seq"],
                max_len=self.enc_max_len,
                device=self._enc_device,
            )

        # Bridge
        h_z = self.W_align(z_enc)

        # Losses auxiliares
        reg_mask = batch["reg_y"] >= 0
        cls_mask = batch["bin_y"] >= 0

        reg_loss = (
            self.huber(reg_pred[reg_mask], batch["reg_y"][reg_mask])
            if reg_mask.any()
            else lm_loss * 0.0
        )
        
        cls_loss = (
            self.bce(cls_log[cls_mask], batch["bin_y"][cls_mask])
            if cls_mask.any()
            else lm_loss * 0.0
        )

        align_mse = F.mse_loss(h_z, h_T)

        # ============================================================
        # Regularización de z_norm (fuerza uso del latente)
        # ============================================================
        z_norm = z_enc.norm(dim=-1).mean()
        target_norm = float(self.hparams.target_znorm)
        znorm_penalty = (z_norm - target_norm).abs()

        # ============================================================
        # Loss total
        # ============================================================
        loss = (
            float(self.hparams.w_lm) * lm_loss
            + float(self.hparams.w_reg) * reg_loss
            + float(self.hparams.w_cls) * cls_loss
            + self.lambda_align() * align_mse
            + float(self.hparams.w_znorm) * znorm_penalty
        )

        # ============================================================
        # Métricas básicas (siempre)
        # ============================================================
        cos_sim = F.cosine_similarity(h_z, h_T, dim=-1).mean()

        logs = {
            f"{stage}_loss": loss,
            f"{stage}_lm": lm_loss,
            f"{stage}_reg": reg_loss,
            f"{stage}_cls": cls_loss,
            f"{stage}_align_mse": align_mse,
            f"{stage}_align_cos": cos_sim,
            f"{stage}_z_norm": z_norm,
            f"{stage}_znorm_penalty": znorm_penalty,
            f"{stage}_lambda_align": self.lambda_align(),
        }

        # ============================================================
        # Métricas adicionales (solo validación)
        # ============================================================
        if stage == "val":
            z_std = z_enc.std(dim=0).mean()
            logs[f"{stage}_z_std"] = z_std
            logs[f"{stage}_z_effdim"] = self.participation_ratio(z_enc)

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # ------------------------------------------------------------
    # Optimizer con weight decay + cosine scheduler
    # ------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params, 
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams.max_epochs),
            eta_min=float(self.hparams.lr) * 0.01  # 1% del LR inicial
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--encoder_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01,
                    help="Weight decay para AdamW (regularización L2)")
    ap.add_argument("--zdim", type=int, default=64)

    ap.add_argument("--w_lm", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.3)
    ap.add_argument("--w_cls", type=float, default=0.5)
    ap.add_argument("--w_align", type=float, default=0.3)
    
    # Regularización z_norm
    ap.add_argument("--w_znorm", type=float, default=0.01,
                    help="Peso de la regularización de z_norm")
    ap.add_argument("--target_znorm", type=float, default=8.0,
                    help="Norma objetivo para z (sqrt(zdim) es buen default)")

    # LoRA config (nuevos defaults anti-overfitting)
    ap.add_argument("--lora_r", type=int, default=4,
                    help="LoRA rank (reducido de 8 a 4)")
    ap.add_argument("--lora_alpha", type=int, default=8,
                    help="LoRA alpha (reducido proporcionalmente)")
    ap.add_argument("--lora_dropout", type=float, default=0.15,
                    help="LoRA dropout (aumentado de 0.05 a 0.15)")

    # Early stopping
    ap.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (reducido de 20 a 10)")

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", default="bf16-mixed")

    args = ap.parse_args()
    seed_all(42)

    # ============================================================
    # Crear estructura de directorios
    # ============================================================
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    art_dir = os.path.join(args.out_dir, "artifacts")
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print("[INFO] CSV:", args.csv)
    print("[INFO] Encoder ckpt:", args.encoder_ckpt)
    print("[INFO] OUT:", args.out_dir)
    print(f"[INFO] Estructura: checkpoints/ logs/ artifacts/")
    print(f"[INFO] Anti-overfitting config:")
    print(f"       - lora_r: {args.lora_r}")
    print(f"       - lora_dropout: {args.lora_dropout}")
    print(f"       - weight_decay: {args.weight_decay}")
    print(f"       - patience: {args.patience}")
    print(f"       - cosine LR schedule")

    # ============================================================
    # Dataset con split por SECUENCIA
    # ============================================================
    base = LongCSV(args.csv)

    seq2idx = {}
    for i, (seq, bug, y) in enumerate(base.data):
        seq2idx.setdefault(seq, []).append(i)

    all_seqs = list(seq2idx.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_seqs)

    n_tr = int(0.9 * len(all_seqs))
    train_seqs = set(all_seqs[:n_tr])
    val_seqs = set(all_seqs[n_tr:])

    train_idx, val_idx = [], []
    for seq, idxs in seq2idx.items():
        if seq in train_seqs:
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs)

    tr = torch.utils.data.Subset(base, train_idx)
    va = torch.utils.data.Subset(base, val_idx)

    print(f"[INFO] Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")
    print(f"[INFO] Train filas: {len(tr)}, Val filas: {len(va)}")

    # ============================================================
    # Model
    # ============================================================
    model = LitFTLatent2LoRA(
        encoder_ckpt=args.encoder_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        zdim=args.zdim,
        w_lm=args.w_lm,
        w_reg=args.w_reg,
        w_cls=args.w_cls,
        w_align=args.w_align,
        w_znorm=args.w_znorm,
        target_znorm=args.target_znorm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_epochs=args.epochs,
    )

    # ============================================================
    # DataLoaders
    # ============================================================
    dl_tr = DataLoader(
        LMCondDataset(tr, model.tok, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    dl_va = DataLoader(
        LMCondDataset(va, model.tok, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ============================================================
    # Logger (CSVLogger → logs/phase1C_lora/version_X/metrics.csv)
    # ============================================================
    logger = CSVLogger(
        save_dir=os.path.join(args.out_dir, "logs"),
        name="phase1C_lora",
    )

    # ============================================================
    # Callbacks
    # ============================================================
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch:03d}-val_lm={val_lm:.3f}",
        monitor="val_lm",
        mode="min",
        save_top_k=2,
        save_last=True,
    )

    # Reanudación
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if not os.path.isfile(resume_ckpt):
        resume_ckpt = None
    else:
        print(f"[INFO] Reanudando desde: {resume_ckpt}")

    # ============================================================
    # Trainer
    # ============================================================
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=[
            ckpt_cb,
            EarlyStopping("val_lm", patience=args.patience, mode="min"),
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dl_tr, dl_va, ckpt_path=resume_ckpt)

    # ============================================================
    # Guardado de artefactos
    # ============================================================
    model.tok.save_pretrained(os.path.join(art_dir, "tokenizer"))
    
    # Guardar LoRA adapter
    model.dec.save_pretrained(os.path.join(art_dir, "lora_adapter"))
    
    # State dict completo
    torch.save(
        {"state_dict": model.state_dict(), "hparams": dict(model.hparams)},
        os.path.join(art_dir, "model_full.pt"),
    )

    best_ckpt = ckpt_cb.best_model_path
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    print(f"[INFO] Logs en: {logger.log_dir}")
    print(f"[OK] Entrenamiento LoRA completado. Artefactos en: {art_dir}")


if __name__ == "__main__":
    main()
