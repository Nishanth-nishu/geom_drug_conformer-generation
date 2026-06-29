# ============================================================
# Dockerfile — mol_next_gen: Molecular Conformer Generation
#
# Docker Hub: nishanthr23/nextmol
# Pull:       docker pull nishanthr23/nextmol:latest
#
# Build context: project root (mol_next_gen/)
# Usage:
#   # Run Experiment F (heavy-atom, DDP):
#   docker run --gpus all --rm \
#     -v /scratch/nishanth.r/nextmol_experiment/mol_next_gen/data:/workspace/data \
#     -v /scratch/nishanth.r/nextmol_experiment/mol_next_gen/experiments:/workspace/experiments \
#     nishanthr23/nextmol:latest \
#     torchrun --nproc_per_node=2 autoresearch/mol_train_ddp.py --exp F
#
#   # Interactive shell:
#   docker run --gpus all -it --rm \
#     -v $(pwd)/data:/workspace/data \
#     nishanthr23/nextmol:latest bash
#
# Base image: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
#   - Matches the cluster's CUDA 12.1 environment exactly
#   - PyTorch 2.5.1+cu121 binaries match
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="nishanthr23"
LABEL description="mol_next_gen: Equivariant Diffusion for Molecular Conformer Generation"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3.10 is the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ── Python + PyTorch (cu121) ──────────────────────────────────────────────────
# Install pip and upgrade
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch separately (CUDA 12.1 wheel index)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric
RUN pip install --no-cache-dir torch-geometric==2.7.0

# Install RDKit (pip wheel, no conda needed)
RUN pip install --no-cache-dir rdkit==2026.3.1

# Install remaining dependencies
RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    scipy==1.15.3 \
    matplotlib==3.10.9 \
    pandas==2.3.3 \
    networkx==3.4.2 \
    selfies==2.2.0 \
    tqdm==4.67.3 \
    seaborn==0.13.2 \
    psutil==7.2.2 \
    requests==2.33.1

# ── Project code ──────────────────────────────────────────────────────────────
WORKDIR /workspace

# Copy project code (not data — data is mounted at runtime)
COPY autoresearch/ ./autoresearch/
COPY models/ ./models/
COPY data/prepare_qm9_heavy.py ./data/prepare_qm9_heavy.py
COPY scripts/ ./scripts/
COPY requirements.txt ./

# Create data and experiments directories (will be bind-mounted)
RUN mkdir -p data experiments logs

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV NCCL_DEBUG=WARN
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=all

# ── Smoke test at build time ───────────────────────────────────────────────────
# Verify key imports work (no GPU needed for this)
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); import rdkit; print(f'RDKit {rdkit.__version__}'); import numpy; print(f'NumPy {numpy.__version__}'); from models.conformer_diffusion import ConformerDiffusion; print('ConformerDiffusion: OK'); from models.attn_conformer_diffusion import AttnConformerDiffusion; print('AttnConformerDiffusion: OK'); print('ALL IMPORTS OK')"

# ── Default command ───────────────────────────────────────────────────────────
# Override with any of:
#   docker run ... nishanthr23/nextmol python autoresearch/mol_train_ddp.py --exp F --no-ddp
#   docker run ... nishanthr23/nextmol torchrun --nproc_per_node=2 autoresearch/mol_train_ddp.py --exp F
CMD ["python3", "autoresearch/mol_train_ddp.py", "--exp", "F", "--no-ddp"]
