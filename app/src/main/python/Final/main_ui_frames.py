# main_ui_frames.py

import os
import sys
import tempfile
import argparse
import threading
import queue
import time

###############################################################################
# Monkey-patch os.makedirs to redirect attempts to create "/Ultralytics"
###############################################################################
#_original_makedirs = os.makedirs
# def custom_makedirs(path, *args, **kwargs):
#     # If the path starts with "/Ultralytics", redirect it to HOME/Ultralytics.
#     if path.startswith("/Ultralytics") or path.startswith("/tmp/Ultralytics"):
#         new_path = os.path.join(os.environ.get("HOME", os.getcwd()), path.lstrip("/"))
#         print(f"Redirecting makedirs from {path} to {new_path}")
#         return _original_makedirs(new_path, *args, **kwargs)
#     return _original_makedirs(path, *args, **kwargs)
# os.makedirs = custom_makedirs

###############################################################################
# Environment Setup
###############################################################################
def setup_environment(home_dir):

    # Validate home_dir: if it's empty or just '/', fallback to os.getcwd()
    if not home_dir.strip() or home_dir == "/":
        print("Warning: Provided home_dir is empty or '/'. Using os.getcwd() instead.")
        home_dir = os.getcwd()

    # Set the HOME environment variable to the provided home_dir.
    # On Android, Chaquopy sets HOME to a writable location (e.g. /data/data/<app_id>/files).
    os.environ["HOME"] = home_dir
    os.chdir(home_dir)  # Change the current working directory to home_dir.
    print("Current working directory changed to:", os.getcwd())

    # Create a temporary directory within home_dir.
    # This directory will be used by libraries that rely on TMPDIR.
    tmp_dir = os.path.join(home_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print("Temporary directory created at:", tmp_dir)

    # Set the TMPDIR environment variable to this temporary directory.
    os.environ["TMPDIR"] = tmp_dir



###############################################################################
# Detection and Processing Functions
###############################################################################
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def process_frame(frame, q):
    """
    Processes a single frame using Ultralytics and puts audio feedback info in the queue.
    """



    # Import Ultralytics-dependent modules after environment setup.
    from Final.object_and_depth_estimation import model, category_index, get_depth_map, get_object_depth_k_median, proximity_level_scene
    import cv2

    depth_map = get_depth_map(frame)
    print("///////////////// started process frame ///////////////")
    detections = []
    results = model(frame)
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()
        conf = result.conf.cpu().numpy()
        cls = int(result.cls.cpu().numpy().item())
        if conf > 0.75:
            startX, startY, endX, endY = box.astype(int)
            width = frame.shape[1]
            left_threshold = width // 3
            right_threshold = 2 * width // 3
            left_overlap = max(0, min(endX, left_threshold) - startX)
            center_overlap = max(0, min(endX, right_threshold) - max(startX, left_threshold))
            right_overlap = max(0, endX - max(startX, right_threshold))
            overlaps = {'left': left_overlap, 'center': center_overlap, 'right': right_overlap}
            section = max(overlaps, key=overlaps.get)
            object_depth = get_object_depth_k_median(depth_map, (startX, startY, endX, endY))
            prox_level = proximity_level_scene(object_depth, depth_map) if object_depth is not None else "Far"
            detection = {
                "box": (startX, startY, endX, endY),
                "label": category_index[cls],
                "section": section,
                "depth": object_depth,
                "prox_level": prox_level
            }
            detections.append(detection)
            depth_text = f"Depth: {object_depth:.2f} ({prox_level})" if object_depth is not None else "Depth: N/A"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            label_text = f'{category_index[cls]}: {section}, {depth_text}'
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {category_index[cls]} in {section} with {depth_text}")

    filtered_detections = []
    overlap_threshold = 0.3
    for i, det in enumerate(detections):
        discard = False
        for j, other_det in enumerate(detections):
            if i == j:
                continue
            if compute_iou(det["box"], other_det["box"]) > overlap_threshold:
                if det["depth"] is not None and other_det["depth"] is not None:
                    if det["depth"] > other_det["depth"]:
                        discard = True
                        break
        if not discard:
            filtered_detections.append(det)

    audio_objects = []
    for det in filtered_detections:
        if det["prox_level"] in ("Very Close", "Close"):
            audio_objects.append({
                "label": det["label"],
                "distance": det["prox_level"],
                "location": det["section"]
            })

    if audio_objects:
        q.put({"objects": audio_objects})

def detection_loop_folder(q, folder_path):
    """
    Continuously monitors the given folder for image files,
    processes the oldest one, and then deletes it.
    """
    import cv2
    print("Starting detection loop on folder:", folder_path)
    while True:
        frame_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
        if not frame_files:
            time.sleep(0.1)
            continue
        frame_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        oldest_frame = os.path.join(folder_path, frame_files[0])
        frame = cv2.imread(oldest_frame)
        if frame is None:
            print(f"Could not read {oldest_frame}. Removing it.")
            os.remove(oldest_frame)
            continue
        #print("TWATAs " + frame_files)
        process_frame(frame, q)
        os.remove(oldest_frame)

###############################################################################
# 3. Top-Level Entry Point (Using Threads)
###############################################################################
def initialize_and_run(*argv):
    """
    Top-level function to initialize the environment and run detection and audio feedback.
    Expected arguments:
      --folder : Path to the folder containing saved frames.
      --fps    : Frames per second (unused in this demo).
      --home   : Writable home directory for config and temp files.
    """
    parser = argparse.ArgumentParser(description="Detection with environment setup using threads")
    parser.add_argument("--folder", type=str, default="./buffer", help="Path to the folder containing saved frames.")
    parser.add_argument("--fps", type=str, default="2", help="Frames per second (unused in this demo).")
    parser.add_argument("--home", type=str, required=True, help="Writable home directory for config and temp files.")
    args = parser.parse_args(argv)

    print("Arguments received:", args)


    # Create a thread-safe queue.
    q = queue.Queue()

    # Start the detection loop in a separate thread.
    detection_thread = threading.Thread(target=detection_loop_folder, args=(q, args.folder), daemon=True)
    detection_thread.start()

    # Start the audio feedback thread.
    try:
        from .audio_feedback import generate_audio_feedback
    except Exception as e:
        print("Error importing generate_audio_feedback:", e)
        generate_audio_feedback = lambda q: print("Audio feedback not available.")
    audio_thread = threading.Thread(target=generate_audio_feedback, args=(q,), daemon=True)
    audio_thread.start()

    # Wait for threads (these run indefinitely in this demo).
    detection_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    initialize_and_run(*sys.argv[1:])
