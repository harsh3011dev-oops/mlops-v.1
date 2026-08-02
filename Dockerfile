# =============================================================
# Stage 1: Builder
# Install all dependencies in a separate stage to keep the
# final image small and clean (industry best practice)
# =============================================================
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching — only reinstalls
# packages when requirements.txt actually changes)
COPY requirements.txt .

# Install Python dependencies into a separate folder
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# =============================================================
# Stage 2: Final Runtime Image
# Minimal, secure, production-ready image
# =============================================================
FROM python:3.11-slim AS runtime

# Labels — industry standard metadata
LABEL maintainer="harsh3011dev"
LABEL project="house-price-mlops"
LABEL version="5.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # MLflow 3.x: Use SQLite backend (no server required in container)
    MLFLOW_TRACKING_URI=sqlite:///mlflow-data/mlflow.db \
    # Ensure Python finds our src/ module
    PYTHONPATH=/app

# Create a non-root user for security (never run containers as root)
RUN useradd --create-home --shell /bin/bash mlops_user

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# -------------------------------------------
# Copy all application source files
# -------------------------------------------

# Core application scripts
COPY train.py .
COPY predict_app.py .

# Modular source package (CRITICAL: required for pipeline to run)
COPY src/ ./src/

# Web UI frontend
COPY static/ ./static/

# Training dataset
COPY data/ ./data/

# Trained model artifact (must run train.py locally first to generate)
COPY model/ ./model/

# -------------------------------------------
# Create runtime directories and set ownership
# -------------------------------------------
RUN mkdir -p /app/mlflow-data /app/artifacts && \
    chown -R mlops_user:mlops_user /app

# Switch to non-root user
USER mlops_user

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker daemon restarts container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: Run the FastAPI prediction server
CMD ["uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
