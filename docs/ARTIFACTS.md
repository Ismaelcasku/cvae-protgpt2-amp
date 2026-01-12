# Artifacts and encoder roles

This project has two conceptually different encoder roles.

## 1) Standalone encoder
Produced by: `scripts/encoder_standalone.py`

Used in:
- latent workflow (`latent_train.py` / `latent_generate.py`)
- LoRA workflow (`lora_train.py` / `lora_generate.py`)

Rationale: the standalone encoder provides sequence representations for workflows where conditioning is not learned purely end-to-end.

## 2) End-to-end FullVAE encoder
Produced by:
- `scripts/fullvae_train.py` (non-gated)
- `scripts/fullvae_gate_train.py` (gated)

Used in:
- `scripts/fullvae_generate.py`
- `scripts/fullvae_gate_generate.py`

Rationale: in FullVAE variants the encoder is trained jointly with conditioning/generation components and must be used consistently.

## Typical artifact bundle (expected by generators)
Most generators expect an `artifacts_dir/` containing a subset of:
- `metadata.json`
- `tokenizer/`
- `encoder.pt` (or equivalent exported encoder)
- `prefix_mlp.pt` (if prefix conditioning is used)
- `lora_adapter/` (if LoRA is used)
- `gate.pt` (gated FullVAE only)
- `heads.pt` (optional downstream predictors, e.g., MIC/activity heads)

Exact filenames may vary by experiment; use the metadata file and script arguments as the source of truth.
