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

# Wait for MLflow server to start
sleep 15

echo "Running MLflow model registration..."
python /mlflow_server/server.py

# Keep the container running
echo "MLflow setup complete. Server is running..."
wait


# #!/bin/bash
# set -e
# # Start MLflow tracking server in background
# mlflow server \
#     --backend-store-uri sqlite:///mlflow.db \
#     --default-artifact-root /mlflow/artifacts \
#     --host 0.0.0.0 \
#     --port 5000 &

# echo "MLflow server started, running custom MLflow script..."
# python /mlflow_server/mlflow.py
