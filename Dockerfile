# ====================================================================
# Multi-stage / Optimized Dockerfile for Bright Future Image Classifier
# Base Image: Official lightweight Python 3.11 Slim
# ====================================================================
FROM python:3.11-slim

# Set environment variables to optimize Python within Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (significantly reduces Docker image size for Render free tier)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy dependencies list and install FastAPI & web serving packages
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.163.2 \
    uvicorn==0.54.0 \
    python-multipart==0.0.34 \
    pydantic==2.14.3 \
    pillow==13.0.0

# Copy application source code and models
COPY src/ /app/src/
COPY models/ /app/models/
COPY README.md /app/

# Expose container port
EXPOSE 8000

# Health check to ensure service is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start Uvicorn ASGI server bound to 0.0.0.0 and dynamic Render $PORT
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
