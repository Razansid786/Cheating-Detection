#!/bin/bash
set -e

echo "Waiting for MLflow to be ready..."
sleep 10

echo "Starting MLflow server..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow_server/mlruns \
    --host 0.0.0.0 \
    --port 5000 &


echo "Running MLflow model registration..."
python /mlflow_server/server.py

# Keep the container running
echo "MLflow setup complete. Server is running..."
wait

