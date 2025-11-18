import mlflow
from ultralytics import YOLO


mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("cheating_detection_experiment")

model="./models/YOLOv8m_best.pt"
load_model=YOLO(model)
with mlflow.start_run():
 
    mlflow.log_artifact(
        load_model,
        artifact_path="cheating-detector_model",
    )
    