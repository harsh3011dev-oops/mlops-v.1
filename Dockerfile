# =============================================================
# Stage 1: Builder
# Install all dependencies in a separate stage
# =============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies into a separate folder
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# =============================================================
# Stage 2: Final Runtime Image
# Minimal, secure, production-ready image
# =============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="harsh3011dev"
LABEL project="house-price-mlops"
LABEL version="5.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow-data/mlflow.db \
    PYTHONPATH=/app

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash mlops_user

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# -------------------------------------------
# Copy all application source code & dataset
# -------------------------------------------
COPY train.py .
COPY predict_app.py .
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/

# Create runtime directories
RUN mkdir -p /app/mlflow-data /app/artifacts /app/model

# Automatically train model during build to generate model/model.pkl
# (Ensures clean git clones build model.pkl seamlessly)
RUN python train.py

# Set folder permissions for non-root user
RUN chown -R mlops_user:mlops_user /app

# Switch to non-root user
USER mlops_user

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command
CMD ["uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
