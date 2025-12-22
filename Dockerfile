FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables for model caching
# Uses /runpod-volume if network volume is attached, otherwise /app/cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_DATASETS_CACHE=/runpod-volume/huggingface/datasets \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    TORCH_HOME=/runpod-volume/torch \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub

# Create cache directories (both local and volume paths)
RUN mkdir -p /app/cache/huggingface /app/cache/torch \
    /runpod-volume/huggingface /runpod-volume/torch

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Upgrade PyTorch to 2.5+ for enable_gqa support in attention
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Model downloads at runtime on first request (~58GB)
# Attach Network Volume (100GB) to /runpod-volume for persistent caching
# First request: slow (downloads model)
# All future requests: fast (model cached on network volume)

# Expose port (optional, for debugging)
EXPOSE 8000

# Run the handler
CMD ["python", "-u", "handler.py"]
