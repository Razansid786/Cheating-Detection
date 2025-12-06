import mlflow
from ultralytics import YOLO
from pathlib import Path

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