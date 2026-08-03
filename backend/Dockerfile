# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables agar PyTorch & Python tidak menulis cache ke direktori terlarang
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/.cache/torch \
    PORT=7860

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Buat user non-root dengan UID 1000 (Standar Wajib Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Salin dependencies list & install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh script & model AI
COPY --chown=user model_service.py .
COPY --chown=user best_model.pth* ./

EXPOSE 7860

# Jalankan Uvicorn server pada port 7860 (atau sesuai environment PORT dari Hugging Face / Docker)
CMD ["sh", "-c", "uvicorn model_service:app --host 0.0.0.0 --port ${PORT:-7860}"]
