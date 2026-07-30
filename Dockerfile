# =============================================================
# Stage 1: Builder
# Install all dependencies in a separate stage to keep the
# final image small and clean (industry best practice)
# =============================================================
FROM python:3.11.15-slim AS builder

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
FROM python:3.11.15-slim AS runtime

# Labels — industry standard metadata
LABEL maintainer="harsh"
LABEL project="house-price-mlops"
LABEL version="2.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow-data/mlflow.db

# Create a non-root user for security (never run containers as root)
RUN useradd --create-home --shell /bin/bash mlops_user

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy project source files
COPY train.py .
COPY train-traces.py .
COPY predict_app.py .
COPY static ./static/
COPY data/ ./data/

# Copy trained model (must run train.py first to generate this)
COPY model/ ./model/

# Create MLflow data directory and give ownership to the non-root user
RUN mkdir -p /app/mlflow-data && \
    chown -R mlops_user:mlops_user /app

# Switch to non-root user
USER mlops_user

# Expose FastAPI port
EXPOSE 8000

# Default command: Run the FastAPI prediction server
CMD ["uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "8000"]
