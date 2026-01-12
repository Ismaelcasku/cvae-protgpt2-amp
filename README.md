# CVAE Conditioning for Antimicrobial Peptide Design (HPC)

HPC-oriented deep learning pipeline (Phase 1A / 1C / 1D) for representation learning and
conditional generation of antimicrobial peptides (AMPs), including ablation-based diagnostics
to verify conditioning usage.

## Structure
- `scripts/legacy/`: original phase scripts (kept for traceability)
- `slurm/`: SLURM submission templates (container-first)
- `containers/`: Singularity/Apptainer notes (images not committed)
- `docs/`: technical report and documentation
- `runs/`, `artifacts/`: outputs (ignored by git)

## HPC usage (minimal)
1) Edit SLURM scripts under `slurm/` (container path, bind mounts, dataset paths).
2) Submit jobs:
   `sbatch slurm/phase1A.slurm`

See `docs/technical_report.pdf` for the methodology and diagnostics.
