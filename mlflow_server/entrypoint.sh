#!/bin/bash

set -e
# Start MLflow tracking server in background
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000 &

echo "MLflow server started, running custom MLflow script..."
python /mlflow/mlflow.py
