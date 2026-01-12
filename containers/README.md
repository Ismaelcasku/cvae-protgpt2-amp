# Containers

This repository is intended to run on HPC with Singularity/Apptainer.

- Keep the `.sif` image outside git (do not commit).
- Bind the repository into the container (e.g., host repo -> /work inside container).
- Use shared HPC storage for datasets, `runs/`, and `artifacts/`.

Edit your execution commands according to your cluster environment.
