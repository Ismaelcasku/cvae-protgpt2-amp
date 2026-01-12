#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1C Latent2: Encoder-Decoder Bridge con alineamiento latente
- Encoder 1A (congelado) → z externo
- ProtGPT2 decoder → h_T
- Bridge: W_align proyecta z → h_T
- Annealing de lambda_align (epochs 0-2: 0, 3-10: ramp, 10+: full)
- Metrics seguros con bf16 (CPU para eigvalsh)

Estructura de salida:
    out_dir/
    ├── logs/version_X/metrics.csv
    ├── checkpoints/
    │   ├── epoch=xxx-val_lm=x.xxx.ckpt
    │   └── last.ckpt
    └── artifacts/
        ├── model_full.pt
        ├── decoder/
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
# Dataset
# ============================================================

class LongCSV(Dataset):
    """CSV largo: una fila por (sequence, bacteria, MIC)"""
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
    """Dataset condicionado a bacteria para language modeling"""
    def __init__(self, base, tok, max_len):
        self.base = base
        self.tok = tok
        self.max_len = max_len
        self.bos = tok.bos_token or "<BOS>"
        self.eos = tok.eos_token or "<EOS>"
        self.pad = tok.pad_token_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        seq, bug, y = self.base[i]

        # Formato: <BACTERIA><BOS>SEQUENCE<EOS>
        text = f"{BACTOK[bug]}{self.bos}{seq}{self.eos}"
        ids = self.tok.encode(text, add_special_tokens=False)
        ids = ids[: self.max_len + 3]

        attn = [1] * len(ids)
        labels = ids.copy()

        # No predecir token bacteria ni BOS
        if len(labels) > 0:
            labels[0] = -100
        if len(labels) > 1:
            labels[1] = -100

        # Padding
        while len(ids) < self.max_len + 3:
            ids.append(self.pad)
            attn.append(0)
            labels.append(-100)

        # Targets: binario (activo si MIC < 4.0) y regresión
        bin_y = -1.0 if y is None else (1.0 if y < 4.0 else 0.0)
        reg_y = -1.0 if y is None else float(y)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attn": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "reg_y": torch.tensor(reg_y, dtype=torch.float32),
            "bin_y": torch.tensor(bin_y, dtype=torch.float32),
            "seq": seq,
        }


def collate(batch):
    """Collate function para DataLoader"""
    out = {}
    for k in batch[0]:
        if k == "seq":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out

# ============================================================
# Encoder 1A loader (importa desde phase1A_v2.py)
# ============================================================

def load_encoder_1A(ckpt_path):
    """
    Carga encoder 1A_v2 desde encoder_state_dict.pt
    Retorna: (encoder, max_len, latent_dim)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" not in ckpt or "hparams" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt_path} no tiene claves esperadas ('state_dict', 'hparams')"
        )

    # Import robusto: buscar phase1A_v2.py en mismo directorio
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        from phase1A_v2 import EncoderVAE
    except ImportError as e:
        raise ImportError(
            "No se pudo importar EncoderVAE desde phase1A_v2.py. "
            "Verifica que phase1A_v2.py está en el directorio codes/"
        ) from e

    h = ckpt["hparams"]
    latent_dim = int(h.get("latent_dim", 64))
    enc_max_len = int(h.get("max_len", 64))

    # Crear encoder y cargar weights
    enc = EncoderVAE(latent_dim=latent_dim)
    enc.load_state_dict(ckpt["state_dict"], strict=True)
    enc.eval()
    
    for p in enc.parameters():
        p.requires_grad = False

    # Sanity check en CPU
    with torch.no_grad():
        z, mu = enc.encode_sequences(["ACDEFGHIK"], max_len=enc_max_len, device="cpu")
        if z.ndim != 2:
            raise RuntimeError(f"encode_sequences devolvió z con shape inesperado: {z.shape}")

    return enc, enc_max_len, latent_dim

# ============================================================
# Lightning Module
# ============================================================

class LitFTLatent2(pl.LightningModule):
    """
    Encoder-Decoder Bridge con alineamiento latente
    """
    def __init__(
        self,
        decoder_dir: str,
        encoder_ckpt: str,
        lr: float,
        w_lm: float,
        w_reg: float,
        w_cls: float,
        w_align: float,
        zdim: int,
        unfreeze_last: int = 2,
        quick_gen_every: int = 0,
        quick_gen_n: int = 16,
        quick_gen_max_new: int = 40,
        **kwargs,  # absorber argumentos extra de main
    ):
        super().__init__()
        self.save_hyperparameters()

        # ============================================================
        # Tokenizer + Decoder (ProtGPT2 base)
        # ============================================================
        print("[INFO] Cargando ProtGPT2 base (safetensors)...")
        
        self.tok = AutoTokenizer.from_pretrained(
            "nferruz/ProtGPT2",
            trust_remote_code=False
        )
        
        self.dec = AutoModelForCausalLM.from_pretrained(
            "nferruz/ProtGPT2",
            use_safetensors=True,
            trust_remote_code=False
        )

        # Añadir tokens bacterianos
        self.tok.add_special_tokens(
            {"additional_special_tokens": list(BACTOK.values())}
        )
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({"pad_token": "<PAD>"})

        self.dec.resize_token_embeddings(len(self.tok))

        # ============================================================
        # Cargar pesos previos del decoder (si existen)
        # ============================================================
        model_pt = os.path.join(decoder_dir, "model.pt")
        if os.path.isfile(model_pt):
            print(f"[INFO] Cargando pesos previos del decoder desde {model_pt}")
            ckpt = torch.load(model_pt, map_location="cpu")
            dec_state = {
                k.replace("dec.", ""): v
                for k, v in ckpt.items()
                if k.startswith("dec.")
            }
            if dec_state:
                self.dec.load_state_dict(dec_state, strict=False)
                print(f"[INFO] Cargados {len(dec_state)} parámetros del decoder")
        else:
            print("[INFO] No se encontraron pesos previos; usando ProtGPT2 base")

        # ============================================================
        # Congelar bloques del decoder
        # ============================================================
        blocks = self.dec.transformer.h
        for i, blk in enumerate(blocks):
            if i < len(blocks) - unfreeze_last:
                for p in blk.parameters():
                    p.requires_grad = False

        print(
            f"[INFO] Congelados {len(blocks) - unfreeze_last}/{len(blocks)} bloques del decoder"
        )

        hid = self.dec.config.n_embd
        self.hid = hid

        # ============================================================
        # Heads de predicción (sobre h_T original)
        # ============================================================
        self.reg_head = nn.Linear(hid, 1)
        self.cls_head = nn.Linear(hid, 1)

        # ============================================================
        # Encoder 1A (congelado, carga en CPU)
        # ============================================================
        enc, enc_max_len, enc_latent_dim = load_encoder_1A(encoder_ckpt)
        
        if enc_latent_dim != zdim:
            raise ValueError(
                f"zdim={zdim} pero encoder latent_dim={enc_latent_dim}. "
                f"Deben coincidir."
            )

        self.encoder_1A = enc
        self.enc_max_len = enc_max_len
        self._enc_device = "cpu"

        print(
            f"[INFO] Encoder 1A cargado: latent_dim={enc_latent_dim}, max_len={enc_max_len}"
        )

        # ============================================================
        # Bridge: z_enc → h_T
        # ============================================================
        self.W_align = nn.Linear(zdim, hid)

        # ============================================================
        # Losses
        # ============================================================
        self.huber = nn.HuberLoss()
        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------
    def on_fit_start(self):
        """Mover encoder al mismo device que el decoder"""
        device = next(self.dec.parameters()).device
        self.encoder_1A = self.encoder_1A.to(device)
        self._enc_device = str(device)

        if self.global_rank == 0:
            print(f"[INFO] Encoder 1A movido a device: {self._enc_device}")

    # ------------------------------------------------------------
    # Metrics helpers (ROBUSTOS: bf16 + batch pequeño + estabilidad)
    # ------------------------------------------------------------
    @staticmethod
    def participation_ratio(z):
        """
        Participation ratio: dimensionalidad efectiva del espacio latente.

        ROBUSTEZ:
        - Convierte a CPU + float32 (eigvalsh no soporta bf16 en CUDA)
        - Maneja batches pequeños
        - Regulariza covarianza para evitar matrices mal condicionadas
        - Retorna NaN si no es computable (Lightning lo ignora)
        """
        with torch.no_grad():
            z_f32 = z.detach().to("cpu", dtype=torch.float32)

            # Batch demasiado pequeño → métrica no fiable
            if z_f32.size(0) < 8:
                return torch.tensor(float("nan"))

            try:
                # Covarianza
                cov = torch.cov(z_f32.T)

                # Regularización diagonal para estabilidad numérica
                eps = 1e-6
                cov = cov + eps * torch.eye(
                    cov.size(0), device=cov.device, dtype=cov.dtype
                )

                # Eigenvalues
                eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-10)

                # Participation ratio
                pr = (eigs.sum() ** 2) / (eigs ** 2).sum()
                return pr

            except (RuntimeError, torch._C._LinAlgError):
                # Fallo numérico → no romper entrenamiento
                return torch.tensor(float("nan"))

    @staticmethod
    def linear_cka(X, Y):
        """
        Linear CKA: similitud entre representaciones.

        ROBUSTEZ:
        - CPU + float32
        - Maneja batches pequeños
        - Protegido contra errores numéricos
        """
        with torch.no_grad():
            # Batch demasiado pequeño → métrica no fiable
            if X.size(0) < 8:
                return torch.tensor(float("nan"))

            X = X.detach().to("cpu", dtype=torch.float32)
            Y = Y.detach().to("cpu", dtype=torch.float32)

            try:
                # Centrar
                X = X - X.mean(0, keepdim=True)
                Y = Y - Y.mean(0, keepdim=True)

                # HSIC lineal
                hsic = torch.trace(X.T @ X @ Y.T @ Y)

                norm_x = torch.norm(X.T @ X, p="fro")
                norm_y = torch.norm(Y.T @ Y, p="fro")

                cka = hsic / (norm_x * norm_y + 1e-10)
                return cka

            except RuntimeError:
                return torch.tensor(float("nan"))

    # ------------------------------------------------------------
    # Annealing de lambda_align
    # ------------------------------------------------------------
    def lambda_align(self):
        """
        Annealing schedule para w_align:
        - Epochs 0-2: 0.0 (warm-up)
        - Epochs 3-10: ramp linealmente de 0 a w_align
        - Epochs 10+: w_align completo
        """
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
        """Forward pass del decoder"""
        return self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )

    # ------------------------------------------------------------
    # Training/validation step
    # ------------------------------------------------------------
    def _step(self, batch, stage: str):
        """Step compartido para train/val"""
        # Forward decoder
        out = self(
            batch["input_ids"],
            batch["attn"],
            batch["labels"],
        )
        lm_loss = out.loss

        # Extraer h_T (último token con atención)
        last = out.hidden_states[-1]  # [B, L, H]
        idx = batch["attn"].sum(dim=1).clamp(min=1) - 1
        h_T = last[torch.arange(last.size(0), device=last.device), idx]  # [B, H]

        # Predicciones de heads (sobre h_T original, NO degradado)
        reg_pred = self.reg_head(h_T).squeeze(-1)
        cls_log = self.cls_head(h_T).squeeze(-1)

        # ============================================================
        # Encoder 1A: obtener z externo (congelado)
        # ============================================================
        with torch.no_grad():
            z_enc, mu_enc = self.encoder_1A.encode_sequences(
                batch["seq"],
                max_len=self.enc_max_len,
                device=self._enc_device,
            )

        # Proyectar z → h
        h_z = self.W_align(z_enc)

        # ============================================================
        # Losses auxiliares
        # ============================================================
        reg_mask = batch["reg_y"] >= 0
        cls_mask = batch["bin_y"] >= 0

        reg_loss = (
            self.huber(reg_pred[reg_mask], batch["reg_y"][reg_mask])
            if reg_mask.any()
            else lm_loss * 0.0  # mantener computation graph
        )
        
        cls_loss = (
            self.bce(cls_log[cls_mask], batch["bin_y"][cls_mask])
            if cls_mask.any()
            else lm_loss * 0.0
        )

        # Alignment loss
        align_mse = F.mse_loss(h_z, h_T)

        # ============================================================
        # Loss total (con annealing de align)
        # ============================================================
        loss = (
            float(self.hparams.w_lm) * lm_loss
            + float(self.hparams.w_reg) * reg_loss
            + float(self.hparams.w_cls) * cls_loss
            + self.lambda_align() * align_mse
        )

        # ============================================================
        # Métricas básicas (siempre)
        # ============================================================
        cos_sim = F.cosine_similarity(h_z, h_T, dim=-1).mean()
        z_std = z_enc.std(dim=0).mean()
        
        logs = {
            f"{stage}_loss": loss,
            f"{stage}_lm": lm_loss,
            f"{stage}_reg": reg_loss,
            f"{stage}_cls": cls_loss,
            f"{stage}_align_mse": align_mse,
            f"{stage}_align_cos": cos_sim,
            f"{stage}_z_std": z_std,
            f"{stage}_lambda_align": self.lambda_align(),
        }

        # ============================================================
        # Métricas pesadas (solo en validación)
        # ============================================================
        if stage == "val":
            logs[f"{stage}_cka"] = self.linear_cka(h_z, h_T)
            logs[f"{stage}_z_effdim"] = self.participation_ratio(z_enc)

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    def configure_optimizers(self):
        """Solo optimizar parámetros trainables"""
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=float(self.hparams.lr))

# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV de entrenamiento")
    ap.add_argument("--encoder_ckpt", required=True, help="Checkpoint encoder 1A")
    ap.add_argument("--decoder_dir", required=True, help="Directorio con decoder base/previo")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida")
    
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--unfreeze_last", type=int, default=2)
    
    ap.add_argument("--w_lm", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=0.3)
    ap.add_argument("--w_cls", type=float, default=0.5)
    ap.add_argument("--w_align", type=float, default=0.3)
    
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    
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
    print("[INFO] Decoder dir:", args.decoder_dir)
    print("[INFO] Encoder ckpt:", args.encoder_ckpt)
    print("[INFO] OUT:", args.out_dir)
    print(f"[INFO] Estructura: checkpoints/ logs/ artifacts/")

    # ============================================================
    # Dataset (SPLIT CORRECTO POR SECUENCIA – SIN LEAKAGE)
    # ============================================================
    base = LongCSV(args.csv)

    # 1) Agrupar índices por SECUENCIA
    seq2idx = {}
    for i, (seq, bug, y) in enumerate(base.data):
        seq2idx.setdefault(seq, []).append(i)

    all_seqs = list(seq2idx.keys())

    # 2) Shuffle determinista de secuencias
    rng = np.random.default_rng(42)
    rng.shuffle(all_seqs)

    # 3) Split 90 / 10 A NIVEL DE SECUENCIA
    n_tr = int(0.9 * len(all_seqs))
    train_seqs = set(all_seqs[:n_tr])
    val_seqs   = set(all_seqs[n_tr:])

    # 4) Reconstruir índices SIN compartir secuencias
    train_idx = []
    val_idx = []

    for seq, idxs in seq2idx.items():
        if seq in train_seqs:
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs)

    tr = torch.utils.data.Subset(base, train_idx)
    va = torch.utils.data.Subset(base, val_idx)

    print("[INFO] Split por secuencia completado")
    print(f"       Secuencias totales : {len(all_seqs)}")
    print(f"       Train seqs         : {len(train_seqs)}")
    print(f"       Val seqs           : {len(val_seqs)}")
    print(f"       Train filas        : {len(tr)}")
    print(f"       Val filas          : {len(va)}")

    # ============================================================
    # Model
    # ============================================================
    model = LitFTLatent2(
        decoder_dir=args.decoder_dir,
        encoder_ckpt=args.encoder_ckpt,
        lr=args.lr,
        w_lm=args.w_lm,
        w_reg=args.w_reg,
        w_cls=args.w_cls,
        w_align=args.w_align,
        zdim=args.zdim,
        unfreeze_last=args.unfreeze_last,
    )

    tok = model.tok

    # ============================================================
    # DataLoaders
    # ============================================================
    dl_tr = DataLoader(
        LMCondDataset(tr, tok, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dl_va = DataLoader(
        LMCondDataset(va, tok, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ============================================================
    # Logger (CSVLogger → logs/version_X/metrics.csv)
    # ============================================================
    logger = CSVLogger(
        save_dir=os.path.join(args.out_dir, "logs"),
        name="phase1C_latent",
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

    # ------------------------------------------------------------
    # Reanudar desde checkpoint si existe (robusto a walltime)
    # ------------------------------------------------------------
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if not os.path.isfile(resume_ckpt):
        resume_ckpt = None
    else:
        print(f"[INFO] Reanudando entrenamiento desde: {resume_ckpt}")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=[
            ckpt_cb,
            EarlyStopping(monitor="val_lm", patience=15, mode="min"),
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    # ============================================================
    # Training (con reanudación automática si existe last.ckpt)
    # ============================================================
    trainer.fit(model, dl_tr, dl_va, ckpt_path=resume_ckpt)

    # ============================================================
    # Guardado de artefactos
    # ============================================================
    # Tokenizer
    model.tok.save_pretrained(os.path.join(art_dir, "tokenizer"))
    
    # Decoder (formato HuggingFace)
    model.dec.save_pretrained(os.path.join(art_dir, "decoder"))
    
    # State dict completo (para cargar todo: heads + W_align)
    torch.save(
        {"state_dict": model.state_dict(), "hparams": dict(model.hparams)},
        os.path.join(art_dir, "model_full.pt"),
    )

    best_ckpt = ckpt_cb.best_model_path
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    print(f"[INFO] Logs en: {logger.log_dir}")
    print(f"[OK] Entrenamiento completado. Artefactos en: {art_dir}")

if __name__ == "__main__":
    main()
