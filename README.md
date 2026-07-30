# 🏡 House Price MLOps Project

A complete end-to-end **MLOps pipeline** for predicting house prices using Machine Learning, deployed on Kubernetes with a FastAPI backend and a modern web UI.

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | RandomForestRegressor (scikit-learn) |
| **Experiment Tracking** | MLflow 3.x with PostgreSQL 16 backend |
| **Tracing** | MLflow Tracing (`@mlflow.trace` decorators) |
| **API Server** | FastAPI + Uvicorn |
| **Frontend UI** | HTML/CSS/JS (Indian Rupee ₹ currency support) |
| **Containerization** | Docker (multi-stage build) |
| **Orchestration** | Kubernetes (Kind cluster) |
| **Database** | PostgreSQL 16 (MLflow metadata backend) |

---

## 📁 Project Structure

```
mlops-v.1/
│
├── 📄 train.py                   # Model training + MLflow experiment logging
├── 📄 train-traces.py            # MLflow Tracing pipeline (step-by-step spans)
├── 📄 predict_app.py             # FastAPI prediction server
│
├── 📁 static/
│   └── index.html               # Web UI (Indian ₹ currency format)
│
├── 📁 k8s/                      # Kubernetes manifests
│   ├── predict-deployment.yaml  # FastAPI pod deployment
│   ├── predict-service.yaml     # K8s Service (port 8000)
│   ├── deployment.yaml          # MLflow UI deployment
│   └── service.yaml             # MLflow UI service
│
├── 📄 Dockerfile                # Multi-stage Docker build
├── 📄 requirements.txt          # Python dependencies
└── 📄 README.md                 # This file
```

---

## ⚡ Quick Start

### 1. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start PostgreSQL (Docker)
```bash
docker run -d \
  --name postgres16 \
  -p 5432:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mlflowdb \
  postgres:16
```

### 3. Start MLflow Tracking Server
```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql+psycopg2://postgres:postgres@localhost:5432/mlflowdb \
  --default-artifact-root ./mlruns \
  --serve-artifacts
```

### 4. Train the Model
```bash
python train.py
```

### 5. Run Training with Traces
```bash
python train-traces.py
```

### 6. Run FastAPI Locally
```bash
uvicorn predict_app:app --host 0.0.0.0 --port 8000
```

---

## ☸️ Kubernetes Deployment (Kind)

### Create Kind Cluster
```bash
kind create cluster --name mlops-cluster
```

### Build & Load Docker Image
```bash
docker build -t houseprice-estimator:v5 .
kind load docker-image houseprice-estimator:v5 --name mlops-cluster
```

### Deploy to Kubernetes
```bash
kubectl apply -f k8s/predict-deployment.yaml
kubectl apply -f k8s/predict-service.yaml
```

### Access Web UI
```bash
kubectl port-forward service/houseprice-predict-service 8000:8000
```
Open: `http://localhost:8000`

---

## 📊 Model Features

| Feature | Description |
|---|---|
| `LotArea` | Lot area in square feet |
| `OverallQual` | Overall quality rating (1-10) |
| `OverallCond` | Overall condition rating (1-10) |
| `YearBuilt` | Year of construction |
| `GrLivArea` | Above ground living area (sq ft) |
| `GarageCars` | Garage capacity (number of cars) |

**Target**: `SalePrice` → Predicted in ₹ Rupees (Lakhs / Crores)

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **R² Score** | 0.9803 (98.03% accuracy) |
| **MSE** | 563,107,626.29 |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/predict` | POST | House price prediction |
| `/health` | GET | Kubernetes health check |
| `/docs` | GET | Swagger UI documentation |
