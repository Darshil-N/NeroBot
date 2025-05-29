import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import serial
import time
import threading

# ---------- Arduino Setup ----------
arduino = None
try:
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("Arduino connected.")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")

def send_command(command):
    if arduino and arduino.is_open:
        arduino.write((command + '\n').encode())
        print(f"Sent to Arduino: {command}")
    else:
        print("Arduino not connected.")

# Send initial "trash" signal
send_command("trash")

# ---------- GUI Setup ----------
def launch_gui():
    gui = tk.Tk()
    gui.title("Arduino Servo Control")

    open_button = tk.Button(gui, text="Open", width=20, height=2, command=lambda: send_command("open"))
    close_button = tk.Button(gui, text="Close", width=20, height=2, command=lambda: send_command("close"))
    trash_button = tk.Button(gui, text="Trash", width=20, height=2, command=lambda: send_command("trash"))
    dump_button = tk.Button(gui, text="Dump", width=20, height=2, command=lambda: send_command("dump"))

    open_button.grid(row=0, column=0, padx=10, pady=10)
    close_button.grid(row=0, column=1, padx=10, pady=10)
    trash_button.grid(row=1, column=0, padx=10, pady=10)
    dump_button.grid(row=1, column=1, padx=10, pady=10)

    gui.mainloop()

# Launch GUI in a separate thread
threading.Thread(target=launch_gui, daemon=True).start()

# ---------- YOLO + OpenCV ----------
def load_model(model_path):
    try:
        model = torch.jit.load(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(frame, input_size=(640, 640)):
    h, w = frame.shape[:2]
    scale = min(input_size[0] / w, input_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    x_offset, y_offset = (input_size[0] - new_w) // 2, (input_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    img = canvas[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0), (x_offset, y_offset, scale)

def process_predictions(predictions, conf_threshold=0.1, iou_threshold=0.45):
    pred = predictions[0].cpu().numpy()
    boxes = pred[:4].T
    scores = pred[4]
    class_ids = pred[5]

    if scores.max() > 1.0:
        scores /= scores.max()

    mask = scores > conf_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return np.array([])

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0:
        return np.array([])

    class_ids = np.array([1 if int(c) >= 2 else int(c) for c in class_ids])
    detections = np.column_stack((boxes_xyxy[indices], scores[indices], class_ids[indices]))
    return detections

def draw_detections(frame, detections, class_names, x_offset, y_offset, scale):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1 = int((x1 - x_offset) / scale)
        y1 = int((y1 - y_offset) / scale)
        x2 = int((x2 - x_offset) / scale)
        y2 = int((y2 - y_offset) / scale)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

        class_id = int(cls)
        if class_id >= len(class_names): class_id = 1
        class_name = class_names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# ---------- Detection Sequence ----------
def detection_sequence():
    send_command("open")
    time.sleep(5)
    send_command("close")
    time.sleep(5)
    send_command("dump")
    time.sleep(5)
    send_command("open")
    time.sleep(5)
    send_command("close")
    time.sleep(1)
    send_command("trash")

# ---------- Webcam Processing ----------
def process_webcam(model, class_names, conf_threshold=0.1, iou_threshold=0.45):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected.")
        return

    detection_in_progress = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, (x_offset, y_offset, scale) = preprocess_image(frame)
        with torch.no_grad():
            predictions = model(input_tensor)

        detections = process_predictions(predictions, conf_threshold, iou_threshold)
        draw_detections(frame, detections, class_names, x_offset, y_offset, scale)

        # Trigger sequence if plastic bottle is detected (class 0)
        if any(int(det[5]) == 0 for det in detections):
            if not detection_in_progress:
                detection_in_progress = True
                print("Plastic bottle detected. Running automation...")
                threading.Thread(target=lambda: [detection_sequence(), setattr(threading.current_thread(), 'done', True)], daemon=True).start()

        cv2.imshow("YOLOv8 + Arduino Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Main ----------
def main():
    model_path = r'C:\Users\Lenovo\Desktop\Folders\Vision Pipeline\NeroBot\runs\detect\train\weights\best.torchscript'
    class_names = ['PlasticBottle - v1 2025-05-16 2-42pm', 'undefined', 'Class 2', 'Class 3', 'Class 4']

    print("Loading model...")
    model = load_model(model_path)
    if model is None:
        return
    print("Model loaded successfully")

    process_webcam(model, class_names)

    if arduino and arduino.is_open:
        arduino.close()

if __name__ == "__main__":
    main()
