FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from environment.yml + extras needed by diffusion_model.py
RUN pip install --no-cache-dir \
    "numpy>=1.20" \
    "click>=8.0" \
    "pillow>=8.3.1" \
    "scipy>=1.7.1" \
    psutil \
    requests \
    tqdm \
    imageio \
    "imageio-ffmpeg>=0.4.3" \
    pyspng \
    matplotlib

# Set working directory
WORKDIR /experiment
