# Containers

This repository is intended to run on HPC with Singularity/Apptainer.

- Keep the `.sif` image outside git (do not commit).
- SLURM scripts under `slurm/` assume you will bind the repo into the container
  (e.g., host repo -> /work inside container).

Edit paths according to your cluster environment.
