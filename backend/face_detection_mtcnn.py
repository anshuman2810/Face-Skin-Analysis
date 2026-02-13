"""
Real-time Face Detection with MTCNN (Lightweight)
Optimized for RTX 3050 (4GB VRAM)
Python 3.10 + PyTorch + CUDA

This version uses MTCNN which is very memory-efficient
and perfect for 4GB VRAM GPUs.
"""

import cv2
import torch
from facenet_pytorch import MTCNN
import time

class MTCNNFaceDetector:
    def __init__(self, device='cuda'):
        """
        Initialize MTCNN face detector
        
        Args:
            device: 'cuda' for GPU or 'cpu' for CPU
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Initialize MTCNN with optimized settings for RTX 3050
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=True  # Detect all faces
        )
        
        print("MTCNN initialized successfully!")
    
    def detect_faces(self, frame):
        """
        Detect faces and facial landmarks
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            boxes: Face bounding boxes
            probs: Detection probabilities
            landmarks: Facial landmarks (eyes, nose, mouth)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)
        
        return boxes, probs, landmarks
    
    def draw_detections(self, frame, boxes, probs, landmarks):
        """
        Draw bounding boxes and landmarks on frame
        
        Args:
            frame: Image to draw on
            boxes: Face bounding boxes
            probs: Detection probabilities
            landmarks: Facial landmarks
        """
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"Face: {prob:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw facial landmarks
                if landmark is not None:
                    for point in landmark:
                        cv2.circle(frame, tuple(map(int, point)), 2, (255, 0, 0), -1)
        
        return frame


def main():
    """Main function to run live face detection"""
    
    # Initialize detector
    print("Initializing MTCNN face detector...")
    detector = MTCNNFaceDetector(device='cuda')
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nFace Detection Started!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    # FPS tracking
    fps_counter = 0
    fps_display = 0
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect faces
        boxes, probs, landmarks = detector.detect_faces(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, boxes, probs, landmarks)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            start_time = time.time()
        
        # Display info
        num_faces = len(boxes) if boxes is not None else 0
        info_text = f"FPS: {fps_display} | Faces: {num_faces}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show VRAM usage
        if detector.device.type == 'cuda':
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_cached = torch.cuda.memory_reserved() / 1024**3
            vram_text = f"VRAM: {vram_used:.2f}/{vram_cached:.2f} GB"
            cv2.putText(frame, vram_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('MTCNN Face Detection - Press Q to quit, S to save', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"face_detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Clear CUDA cache
    if detector.device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\nFace detection stopped.")
    print(f"Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()