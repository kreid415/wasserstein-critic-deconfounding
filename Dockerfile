FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Ignore interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Basic tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ca-certificates \
    build-essential \
    cmake \
    openssh-client \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# install the right miniforge version based on architecture
RUN ARCH="$(uname -m)" && \
    if [ "$ARCH" = "x86_64" ]; then \
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $ARCH"; exit 1; \
    fi && \
    wget "$URL" -O installer.sh && \
    bash installer.sh -b -p /opt/conda && \
    rm installer.sh



ENV PATH="/opt/conda/bin:$PATH"

# Add environment lock file
COPY conda-lock.yml /tmp/conda-lock.yml

# install conda-lock
RUN mamba install conda-lock=2.5.7

# update mamba
RUN mamba update -n base mamba

# create a conda env
ENV CONDA_ENV=./conda/bin
RUN conda-lock install --name myenv /tmp/conda-lock.yml

RUN echo "source activate myenv" > ~/.bashrc
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# install precommit, ruff, and nbstripout
RUN pip install pre-commit ruff nbstripout

# install PyTorch with CUDA (only for Linux GPU image)
RUN pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# install scladders
RUN pip install scladder

COPY . /
RUN pip install .
