import torch
import cv2
import numpy as np

def load_model(model_path):
    """Load the TorchScript model"""
    try:
        model = torch.jit.load(r'C:\Users\Lenovo\Desktop\Folders\Vision Pipeline\NeroBot\runs\detect\train\weights\best.torchscript')
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(frame, input_size=(640, 640)):
    """Preprocess the image for model input"""
    # Get original image dimensions
    height, width = frame.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(input_size[0] / width, input_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create black canvas
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    
    # Calculate offsets to center the image
    x_offset = (input_size[0] - new_width) // 2
    y_offset = (input_size[1] - new_height) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    # Convert to RGB and normalize
    img = canvas[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    
    return torch.from_numpy(img).unsqueeze(0), (x_offset, y_offset, scale)

def process_predictions(predictions, conf_threshold=0.1, iou_threshold=0.45):
    """Process model predictions and return detections"""
    # Convert predictions to numpy array
    pred = predictions[0].cpu().numpy()  # Shape: [6, 8400]
    
    # Extract boxes, scores, and class IDs
    boxes = pred[:4].T  # [8400, 4] - x, y, w, h
    scores = pred[4]    # [8400] - confidence scores
    class_ids = pred[5] # [8400] - class IDs
    
    print(f"Raw predictions - Number of boxes: {len(boxes)}")
    print(f"Confidence scores range: {scores.min():.3f} to {scores.max():.3f}")
    print(f"Class IDs range: {class_ids.min():.3f} to {class_ids.max():.3f}")
    
    # Normalize confidence scores if they're not in [0,1] range
    if scores.max() > 1.0:
        scores = scores / scores.max()
    
    # Filter boxes based on confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    print(f"After confidence filtering - Number of boxes: {len(boxes)}")
    
    if len(boxes) == 0:
        return np.array([])
    
    # Convert boxes to x1, y1, x2, y2 format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy, scores, conf_threshold, iou_threshold)
    
    print(f"After NMS - Number of boxes: {len(indices)}")
    
    if len(indices) == 0:
        return np.array([])
    
    # Convert class IDs to integers and handle out-of-range values
    class_ids = np.array([1 if int(cls_id) >= 2 else int(cls_id) for cls_id in class_ids])
    
    # Combine results
    detections = np.column_stack((
        boxes_xyxy[indices],
        scores[indices],
        class_ids[indices]
    ))
    
    return detections

def draw_detections(frame, detections, class_names, x_offset, y_offset, scale):
    """Draw bounding boxes and labels on the frame"""
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Convert coordinates back to original image space
        x1 = int((x1 - x_offset) / scale)
        y1 = int((y1 - y_offset) / scale)
        x2 = int((x2 - x_offset) / scale)
        y2 = int((y2 - y_offset) / scale)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(frame.shape[1], x1))
        y1 = max(0, min(frame.shape[0], y1))
        x2 = max(0, min(frame.shape[1], x2))
        y2 = max(0, min(frame.shape[0], y2))
        
        if x2 > x1 and y2 > y1:
            # Convert class ID to integer and handle out-of-range values
            class_id = int(cls)
            if class_id >= len(class_names):
                class_id = 1  # Default to 'undefined'
            class_name = class_names[class_id]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{class_name} {conf:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

def process_image(image_path, model, class_names, conf_threshold=0.1, iou_threshold=0.45):
    """Process a single image"""
    # Read image
    print(f"Reading image from: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    print(f"Image loaded successfully. Shape: {frame.shape}")
    
    # Preprocess image
    input_tensor, (x_offset, y_offset, scale) = preprocess_image(frame)
    print(f"Image preprocessed. Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)
        print(f"Model output shape: {predictions.shape}")
        print(f"Model output min/max values: {predictions.min().item():.3f}, {predictions.max().item():.3f}")
    
    # Process predictions
    detections = process_predictions(predictions, conf_threshold, iou_threshold)
    print(f"Number of detections: {len(detections)}")
    
    if len(detections) > 0:
        print("Detection details:")
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            # Handle class ID safely
            class_id = int(cls)
            if class_id >= len(class_names):
                class_id = 1  # Default to 'undefined'
            class_name = class_names[class_id]
            print(f"Detection {i}: Class={class_name}, Confidence={conf:.3f}, Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Draw detections
    draw_detections(frame, detections, class_names, x_offset, y_offset, scale)
    
    # Display image
    cv2.imshow('Object Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'
    cv2.imwrite(output_path, frame)
    print(f"Result saved to: {output_path}")

def process_webcam(model, class_names, conf_threshold=0.1, iou_threshold=0.45):
    """Process video stream from webcam"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Preprocess frame
        input_tensor, (x_offset, y_offset, scale) = preprocess_image(frame)
        
        # Run inference
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Process predictions
        detections = process_predictions(predictions, conf_threshold, iou_threshold)
        
        # Draw detections
        draw_detections(frame, detections, class_names, x_offset, y_offset, scale)
        
        # Display frame
        cv2.imshow('Object Detection - Webcam', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Model path and class names
    model_path = r'C:\Users\Lenovo\Desktop\Folders\Vision Pipeline\NeroBot\runs\detect\train\weights\best.torchscript'
    class_names = ['PlasticBottle - v1 2025-05-16 2-42pm', 'undefined', 'Class 2', 'Class 3', 'Class 4']
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    if model is None:
        return
    print("Model loaded successfully")
    
    # Ask user for input type
    print("\nChoose input type:")
    print("1. Process single image")
    print("2. Use webcam")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Get image path from user
        image_path = r"C:\Users\Lenovo\Pictures\39.jpg"
        print(f"Processing image: {image_path}")
        process_image(image_path, model, class_names)
    elif choice == "2":
        print("Starting webcam...")
        process_webcam(model, class_names)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
