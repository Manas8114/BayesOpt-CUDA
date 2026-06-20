FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to point python to python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip and install PyTorch
RUN python -m pip install --upgrade pip
RUN pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install numpy scipy pytest setuptools

# Copy source
COPY . /app/

# Build and install the extension
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install -e .

CMD ["python", "demo.py"]
