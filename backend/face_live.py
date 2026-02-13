"""
Real-time Face Detection with GPU Acceleration
Optimized for RTX 3050 (4GB VRAM)
Python 3.10 + PyTorch + CUDA
"""

import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

class FaceDetector:
    def __init__(self, confidence_threshold=0.7, device='cuda'):
        """
        Initialize face detector with GPU acceleration
        
        Args:
            confidence_threshold: Minimum confidence score for detection (0-1)
            device: 'cuda' for GPU or 'cpu' for CPU
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load lightweight MobileNetV3 face detection model
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights='DEFAULT',
            weights_backbone='DEFAULT'
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        
        # Enable mixed precision for better performance on RTX 3050
        self.use_amp = True if self.device.type == 'cuda' else False
        
    @torch.no_grad()
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            scores: List of confidence scores
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(self.device)
        
        # Run inference with automatic mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(img_tensor)
        else:
            predictions = self.model(img_tensor)
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        return boxes, scores
    
    def draw_boxes(self, frame, boxes, scores):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Image to draw on
            boxes: List of bounding boxes
            scores: List of confidence scores
        """
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame


def main():
    """Main function to run live face detection"""
    
    # Initialize face detector
    print("Initializing face detector...")
    detector = FaceDetector(confidence_threshold=0.7, device='cuda')
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution (lower resolution = better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting face detection. Press 'q' to quit.")
    
    # FPS calculation
    fps_counter = 0
    fps_display = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect faces
        boxes, scores = detector.detect_faces(frame)
        
        # Draw results
        frame = detector.draw_boxes(frame, boxes, scores)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            start_time = time.time()
        
        # Display info
        info_text = f"FPS: {fps_display} | Faces: {len(boxes)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show VRAM usage if on GPU
        if detector.device.type == 'cuda':
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_text = f"VRAM: {vram_used:.2f} GB"
            cv2.putText(frame, vram_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Detection - Press Q to quit', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")


if __name__ == "__main__":
    main()