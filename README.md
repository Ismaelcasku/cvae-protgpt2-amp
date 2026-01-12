# Generative VAE integrating ProtGPT2 for Antimicrobial Peptides (HPC)

HPC-oriented deep learning project for representation learning and conditional generation of antimicrobial peptides (AMPs).
The pipeline combines a VAE-style latent variable model with a ProtGPT2 decoder and supports multiple conditioning strategies:
latent conditioning, prefix conditioning, LoRA adaptation, and a gated end-to-end FullVAE variant. The repository also includes
ablation-style diagnostics to validate that conditioning is actually used (not ignored).

Technical report: `docs/technical_report.pdf`

## Repository layout
- `scripts/`: runnable entrypoints (`*_train.py`, `*_generate.py`)
- `docs/`: technical report + artifact/usage documentation
- `containers/`: Singularity/Apptainer notes (images are not committed)
- `runs/`, `artifacts/`: outputs (intentionally ignored by git)

## Entry points
### Training
- `scripts/encoder_standalone.py`: standalone sequence encoder (representation model)
- `scripts/latent_train.py`: latent conditioning training workflow
- `scripts/lora_train.py`: LoRA-based fine-tuning workflow
- `scripts/prefix_train.py`: prefix conditioning workflow
- `scripts/fullvae_train.py`: end-to-end FullVAE training (non-gated)
- `scripts/fullvae_gate_train.py`: end-to-end FullVAE training (gated)

### Generation
- `scripts/latent_generate.py`: generation for latent-conditioning workflow
- `scripts/lora_generate.py`: generation for LoRA workflow
- `scripts/prefix_generate.py`: generation for prefix workflow
- `scripts/fullvae_generate.py`: end-to-end FullVAE generation (non-gated)
- `scripts/fullvae_gate_generate.py`: end-to-end FullVAE generation (gated)

### Diagnostics / analysis
- `scripts/fullvae_ablation.py`: ablations for conditioning-usage diagnostics (e.g., shuffling/zeroing controls)

## Artifacts and reproducibility
This repository does not include datasets or large model weights. Training scripts are expected to write artifacts
(checkpoints, tokenizer, adapters, metadata) to a user-defined `artifacts_dir/` on shared HPC storage.

See `docs/ARTIFACTS.md` for:
- the artifact bundle expected by each generator
- the distinction between the standalone encoder and the end-to-end FullVAE encoder
