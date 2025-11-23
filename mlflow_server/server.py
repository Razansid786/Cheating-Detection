import mlflow
from ultralytics import YOLO
from pathlib import Path
import os
import subprocess

# PROJECT_ROOT = Path(__file__).parent.parent  # Adjust based on your structure
# MLFLOW_MODELS_DIR = PROJECT_ROOT / "mlflow_server" / "models"
# os.chdir(PROJECT_ROOT)
# original_dir = os.getcwd()

# subprocess.run(
#         ["dvc", "pull", "mlflow_server/models/YOLOv8m_best.pt.dvc"],
#         check=True,
#         capture_output=True,
#         text=True
#     )
    
# subprocess.run(
#         ["dvc", "pull", "mlflow_server/models/YOLOv8n.pt.dvc"],
#         check=True,
#         capture_output=True,
#         text=True
#     )
# os.chdir(original_dir)

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("cheating_detection_experiment")

# Model 1: Cheating Detector
model1 = "models/YOLOv8m_best.pt"
load_model = YOLO(model1)

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("model_type", "YOLOv8m")
    mlflow.log_param("purpose", "cheating_detection")
    
    # Log artifact
    mlflow.log_artifact(
        model1,
        artifact_path="models",
    )
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/models/YOLOv8m_best.pt"
    mlflow.register_model(
        model_uri=model_uri,
        name="cheating-detector_model"
    )
    
    print("✓ Cheating detector model registered")

# Model 2: Person Detector
model2 = "models/YOLOv8n.pt"
load_model = YOLO(model2)

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("model_type", "YOLOv8n")
    mlflow.log_param("purpose", "person_detection")
    
    # Log artifact
    mlflow.log_artifact(
        model2,
        artifact_path="models",
    )
    
    # Register the model
    model_uri = f"runs:/{run.info.run_id}/models/YOLOv8n.pt"
    mlflow.register_model(
        model_uri=model_uri,
        name="person-detector_model"
    )
    
    print("✓ Person detector model registered")

print("All models registered successfully!")