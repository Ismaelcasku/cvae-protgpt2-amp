#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 1A v2 (reutilizable)
Encoder AMP/noAMP con VAE
- Dataset: Database1_encoder (SIN cambios)
- latent_dim configurable (vamos a usar 64)
- Early stopping
- Exporta encoder limpio + metadata para usar en phase1C_latent2
- Incluye helpers: tokenize(), encode_sequences(), load_encoder_1A()
"""

import os, math, argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision,
    BinaryAccuracy, BinaryF1Score
)

# ---------------- CONFIG ----------------
SEED = 42
AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, MASK, CLS, EOS = 0, 1, 2, 3
AA2ID = {a: i + 4 for i, a in enumerate(AA)}
ID2AA = {i + 4: a for i, a in enumerate(AA)}
VOCAB = 4 + len(AA)

FREE_BITS = 0.75
KL_WARMUP_FRAC = 0.4

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
FF_DIM = 1024
DROPOUT = 0.1


# ---------------- DATASET ----------------
class AmpDataset(Dataset):
    def __init__(self, csv, max_len):
        df = pd.read_csv(csv)
        assert "Sequence" in df.columns and "Activity" in df.columns, "CSV debe tener Sequence y Activity"
        self.seqs = df["Sequence"].astype(str).tolist()
        self.y = torch.tensor(
            df["Activity"].astype(str).str.lower().map({"amp": 1, "noamp": 0}).values,
            dtype=torch.float32
        )
        self.max_len = max_len

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        ids, attn = EncoderVAE.tokenize(self.seqs[i], self.max_len)
        return ids, attn, self.y[i], self.seqs[i]


@dataclass
class Collate:
    train: bool
    wd: float = 0.2

    def __call__(self, batch):
        ids = torch.stack([b[0] for b in batch])
        attn = torch.stack([b[1] for b in batch])
        y = torch.stack([b[2] for b in batch])
        seqs = [b[3] for b in batch]
        if self.train and self.wd > 0:
            # no tocar PAD/MASK/CLS/EOS; solo AA (>3)
            mask = (torch.rand_like(ids.float()) < self.wd) & (ids > 3)
            ids = ids.masked_fill(mask, MASK)
        return ids, attn, y, seqs


# ---------------- MODEL ----------------
class EncoderVAE(pl.LightningModule):
    """
    Encoder VAE para AMP/noAMP.
    Reutilización: tokenize() + encode_sequences() permiten usarlo como extractor de latentes
    en phase1C_latent2.
    """

    def __init__(self, latent_dim=64, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()

        self.emb = nn.Embedding(VOCAB, D_MODEL, padding_idx=PAD)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

        self.fc_mu = nn.Linear(D_MODEL, latent_dim)
        self.fc_lv = nn.Linear(D_MODEL, latent_dim)
        self.cls_head = nn.Linear(latent_dim, 1)

        # métricas
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()

    # ---------- Tokenización portable ----------
    @staticmethod
    def tokenize(seq: str, max_len: int):
        seq = str(seq).strip().upper()
        ids = [CLS] + [AA2ID[c] for c in seq if c in AA2ID] + [EOS]
        ids = ids[:max_len]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            padn = max_len - len(ids)
            ids += [PAD] * padn
            attn += [0] * padn
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)

    def forward(self, ids, attn):
        x = self.emb(ids)
        h = self.enc(x, src_key_padding_mask=~attn)
        h0 = self.norm(h[:, 0])  # CLS token summary (dejamos esto por ahora)
        mu = self.fc_mu(h0)
        lv = self.fc_lv(h0).clamp(-5, 5)
        std = torch.exp(0.5 * lv)
        z = mu + torch.randn_like(mu) * std
        logit = self.cls_head(z).squeeze(-1)
        return mu, lv, z, logit

    # ---------- Encode reutilizable ----------
    @torch.no_grad()
    def encode_sequences(self, sequences, max_len=64, device=None, use_mu=True):
        """
        Devuelve (z_or_mu, mu). Por defecto usa mu(x) (determinístico, recomendado para puente).
        - use_mu=True: z_out = mu
        - use_mu=False: z_out = mu + eps*std
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        ids_list, attn_list = [], []
        for s in sequences:
            ids, attn = self.tokenize(s, max_len)
            ids_list.append(ids)
            attn_list.append(attn)

        ids = torch.stack(ids_list, dim=0).to(device)
        attn = torch.stack(attn_list, dim=0).to(device)

        mu, lv, z, _ = self(ids, attn)
        z_out = mu if use_mu else z
        return z_out, mu

    # ---------- Loss helpers ----------
    def _kl_freebits_anneal(self, mu, lv):
        # KL per-dim mean
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp())  # (B, D)
        kl_dim = kl.mean()  # promedio sobre batch y dims
        kl_fb = torch.relu(kl_dim - FREE_BITS)

        warm = max(1, int(self.trainer.max_epochs * KL_WARMUP_FRAC))
        beta = min(1.0, float(self.current_epoch) / float(warm))
        return beta * kl_fb, kl_dim

    def step(self, batch, stage):
        ids, attn, y, _ = batch
        mu, lv, z, logit = self(ids, attn)

        bce = F.binary_cross_entropy_with_logits(logit, y)
        kl_term, kl_dim = self._kl_freebits_anneal(mu, lv)
        loss = bce + kl_term

        p = torch.sigmoid(logit)
        self.auroc.update(p, y.int())
        self.auprc.update(p, y.int())
        self.acc.update(p, y.int())
        self.f1.update(p, y.int())

        # métricas geométricas simples para diagnosticar colapso
        z_std_mean = mu.std(dim=0).mean()
        z_effdim = (mu.std(dim=0) > 1e-2).float().sum()

        self.log_dict({
            f"{stage}_loss": loss,
            f"{stage}_bce": bce,
            f"{stage}_kl_dim": kl_dim,
            f"{stage}_z_std": z_std_mean,
            f"{stage}_z_effdim": z_effdim,
        }, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, b, _): return self.step(b, "train")
    def validation_step(self, b, _): self.step(b, "val")
    def test_step(self, b, _): self.step(b, "test")

    def on_validation_epoch_end(self):
        self.log_dict({
            "val_auroc": self.auroc.compute(),
            "val_auprc": self.auprc.compute(),
            "val_acc": self.acc.compute(),
            "val_f1": self.f1.compute(),
        }, prog_bar=True)
        self.auroc.reset(); self.auprc.reset(); self.acc.reset(); self.f1.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ---------------- LOAD HELPER ----------------
def load_encoder_1A(export_pt_path: str, device="cpu"):
    """
    Carga encoder desde encoder_state_dict.pt exportado por este script.
    Devuelve modelo en eval(), congelado.
    """
    ckpt = torch.load(export_pt_path, map_location=device)
    h = ckpt.get("hparams", {})
    latent_dim = int(h.get("latent_dim", ckpt.get("latent_dim", 64)))

    model = EncoderVAE(latent_dim=latent_dim, lr=3e-4)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    args = ap.parse_args()

    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    def make_loader(split, train):
        path = os.path.join(args.split_dir, f"{split}.csv")
        ds = AmpDataset(path, args.max_len)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=Collate(train=train, wd=0.2),
        )

    model = EncoderVAE(latent_dim=args.latent_dim, lr=3e-4)

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.base_dir, "checkpoints", "phase1A_v2"),
        filename="phase1A_v2-{epoch:02d}-{val_auroc:.3f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=2
    )
    es = EarlyStopping(monitor="val_auroc", patience=10, mode="max", min_delta=0.001)
    lrmon = LearningRateMonitor("epoch")

    logger = CSVLogger(save_dir=os.path.join(args.base_dir, "logs"), name="phase1A_v2")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[ckpt_cb, es, lrmon],
        logger=logger,
        gradient_clip_val=1.0
    )

    dl_tr = make_loader("train", True)
    dl_va = make_loader("val", False)
    dl_te = make_loader("test", False)

    trainer.fit(model, dl_tr, dl_va)
    trainer.test(model, dl_te)

    # ----- EXPORT ENCODER + METADATA -----
    out = os.path.join(args.base_dir, "artifacts", "encoder1A_v2")
    os.makedirs(out, exist_ok=True)

    export_path = os.path.join(out, "encoder_state_dict.pt")
    torch.save({
        "version": "1A_v2",
        "state_dict": model.state_dict(),
        "hparams": {
            "latent_dim": args.latent_dim,
            "max_len": args.max_len,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "vocab_size": VOCAB,
        },
        "vocab": {
            "aa2id": AA2ID,
            "id2aa": ID2AA,
            "PAD": PAD, "MASK": MASK, "CLS": CLS, "EOS": EOS
        }
    }, export_path)

    print(f"✓ Encoder 1A_v2 exportado: {export_path}")

    # ----- VERIFICACIÓN DE EXPORT -----
    print("\n--- Verificando export ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_encoder_1A(export_path, device=device)
    test_seqs = ["RKKRRQRRR", "ACDEFGHIK", "GIGKFLHSAKKFGKAFVGEIMNS"]
    z_out, mu = loaded.encode_sequences(test_seqs, max_len=args.max_len, device=device, use_mu=True)
    print(f"✓ Encoder cargado: z.shape={tuple(z_out.shape)}  mu.shape={tuple(mu.shape)}")
    print(f"  mu[0,:5]={mu[0,:5].detach().float().cpu().numpy()}")


if __name__ == "__main__":
    main()
