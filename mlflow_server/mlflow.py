import mlflow
from ultralytics import YOLO

import subprocess

subprocess.run(["dvc", "pull", "models/YOLOv8m_best.pt.dvc"], check=True)
subprocess.run(["dvc", "pull", "models/YOLOv8n.pt.dvc"], check=True)

mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("cheating_detection_experiment")

model="../models/YOLOv8m_best.pt"
load_model=YOLO(model)
with mlflow.start_run():

    mlflow.log_artifact(
        load_model,
        artifact_path="cheating-detector_model",
    )

model="../models/YOLOv8n.pt"
load_model=YOLO(model)
with mlflow.start_run():

    mlflow.log_artifact(
        load_model,
        artifact_path="person-detector_model",
    )