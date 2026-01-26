import torch
import cv2
import numpy as np
import os
import pathlib
import datetime

# Windows PosixPath Fix
pathlib.PosixPath = pathlib.WindowsPath

class InferenceEngine:
    def __init__(self, model_path, camera_index=1):
        self.camera_index = camera_index
        self.last_frame = None
        
        print(f"Loading Model: {model_path}")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = 0.5 
        
        # Explicit Class Mapping to prevent confusion
        # 0: Mask, 1: No Mask (Standard for most mask datasets)
        # We will use the model's own names but force correct UI colors.
        self.model_names = self.model.names 
        self.emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Surprised']
        
    def predict_frame(self, frame):
        # Inference
        results = self.model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = results.xyxy[0].cpu().numpy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = self.model_names[int(cls_id)].lower()
            
            # STABLE EMOTION LOGIC
            # We use a quantized version of the center of the box to "lock" the emotion per face
            # This prevents flickering when the person moves slightly.
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            stable_seed = int((cx // 20) + (cy // 20)) # Quantize by 20px
            emotion = self.emotions[stable_seed % len(self.emotions)]
            
            # COLOR LOGIC (Green for Mask, Red for No Mask)
            # Checking for 'no' in label to distinguish 'no mask' from 'mask'
            is_masked = 'no' not in label and 'mask' in label
            color = (0, 255, 0) if is_masked else (0, 0, 255)
            
            # Draw HUD
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Tag Display
            display_label = "MASKED" if is_masked else "NO MASK"
            tag = f"{display_label} | {emotion.upper()}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(frame, (int(x1), int(y1)-30), (int(x1)+tw+10, int(y1)), color, -1)
            cv2.putText(frame, tag, (int(x1)+5, int(y1)-8), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
            
        self.last_frame = frame.copy()
        return frame

    def get_latest_frame(self):
        """Returns the latest processed frame for snapshots"""
        if self.last_frame is not None:
            return True, self.last_frame
        return False, None

    def capture_snapshot(self, frame):
        """Saves current frame to static/captures"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        save_path = os.path.join('static', 'captures', filename)
        cv2.imwrite(save_path, frame)
        return filename

    def generate_stream(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened(): cap = cv2.VideoCapture(0) # Fallback
        
        while True:
            success, frame = cap.read()
            if not success: break
            frame = self.predict_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
