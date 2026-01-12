#!/bin/bash
#SBATCH --job-name=phase1A
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

# === EDITA ESTO ===
CONTAINER="/path/to/pytorch_2.1.0-cuda11.8-cudnn8-runtime.sif"
WORKDIR="/path/to/your/workdir/cvae-amp-design"   # ruta del repo visible en nodos
BIND="${WORKDIR}:/work"
# ==================

cd "${WORKDIR}"

module purge || true
module load singularity || module load apptainer || true

singularity exec --nv -B "${BIND}" "${CONTAINER}" bash -lc "
  cd /work
  python scripts/legacy/phase1A_v2.py --help
"
