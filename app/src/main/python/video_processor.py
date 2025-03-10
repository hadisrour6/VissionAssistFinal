import os
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

base_dir = os.path.dirname(__file__)  # Gets the directory of video_processor.py
best_pt_path = os.path.join(base_dir, "runs", "detect", "train7", "weights", "best.pt")

model = YOLO(best_pt_path)  # Load model using absolute path
category_index = model.names

depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def get_depth_map(frame):
    """Convert frame to depth map using Depth-Anything."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    depth_result = depth_estimator(pil_img)
    depth_tensor = depth_result["predicted_depth"]
    depth_array = depth_tensor.detach().cpu().numpy()
    depth_array = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_array

def process_video(input_video_path):
    """Process the video with YOLO and Depth estimation."""
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        return "Error: Cannot open video file."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_dir = "/storage/emulated/0/Android/data/com.example.vissionassistfinal/files"
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, os.path.basename(input_video_path).replace(".mp4", "_processed.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = get_depth_map(frame)

        results = model(frame)
        for result in results[0].boxes:
            box = result.xyxy[0].cpu().numpy()
            conf = result.conf.cpu().numpy()
            cls = int(result.cls.cpu().numpy())

            if conf > 0.6:
                startX, startY, endX, endY = box.astype(int)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label_text = f"{category_index[cls]}"
                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)  # Save processed frame

    cap.release()
    out.release()

    return output_video_path  # Return processed video path

def run_video_processing(input_video):
    """Main function to be called from Android."""
    return process_video(input_video)
