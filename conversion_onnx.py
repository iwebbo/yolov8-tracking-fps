import torch
from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(imgsz=800)
model.export(format="onnx")
model = YOLO("best.onnx")
model.predict(imgsz=800)