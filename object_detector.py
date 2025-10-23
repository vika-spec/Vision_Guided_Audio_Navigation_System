# vision-audio-navigation/src/object_detector.py

import cv2
import numpy as np
from ultralytics import YOLO
from .utils import Config

class ObjectDetector:
    def __init__(self):
        print("Loading YOLOv8 model...")
        self.model = YOLO(Config.YOLO_MODEL)
        self.navigation_classes = {
            'person': 'person', 'car': 'vehicle', 'truck': 'vehicle', 
            'bus': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'bicycle',
            'traffic light': 'traffic light', 'stop sign': 'stop sign',
            'chair': 'chair', 'bench': 'bench', 'cat': 'animal', 
            'dog': 'animal', 'bird': 'animal'
        }
        print("Object detector initialized!")
    
    def detect_objects(self, frame, frame_width):
        """Detect objects in frame and return detection info"""
        results = self.model(frame, conf=Config.YOLO_CONFIDENCE, verbose=False)
        objects_info = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                nav_label = self.navigation_classes.get(label.lower(), 'object')
                
                bbox_height = y2 - y1
                distance = self._calculate_object_distance(bbox_height, nav_label)
                distance_category = self._get_distance_category(distance)
                position = self._get_object_position([x1, y1, x2, y2], frame_width)
                
                object_info = {
                    'type': 'object',
                    'label': nav_label,
                    'distance': distance,
                    'distance_category': distance_category,
                    'position': position,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'priority': Config.OBJECT_PRIORITY.get(nav_label, 2)
                }
                objects_info.append(object_info)
        
        return objects_info
    
    def _calculate_object_distance(self, bbox_height, object_type):
        """Estimate distance to objects"""
        reference_sizes = {
            'person': 1.7, 'vehicle': 1.5, 'bicycle': 1.0, 'animal': 0.5,
            'chair': 1.0, 'bench': 1.0, 'traffic light': 2.0, 
            'stop sign': 2.0, 'object': 0.5, 'default': 1.0
        }
        
        real_height = reference_sizes.get(object_type, reference_sizes['default'])
        focal_length = 500
        
        if bbox_height > 0:
            distance = (focal_length * real_height) / bbox_height
            return max(0.5, min(distance, 20))
        return 20
    
    def _get_object_position(self, bbox, frame_width):
        """Determine object position in frame"""
        x_center = (bbox[0] + bbox[2]) / 2
        third = frame_width / 3
        
        if x_center < third:
            return "left"
        elif x_center < 2 * third:
            return "center"
        else:
            return "right"
    
    def _get_distance_category(self, distance):
        """Convert distance to descriptive category"""
        if distance < 2:
            return "very close"
        elif distance < 4:
            return "close"
        elif distance < 7:
            return "moderate distance"
        elif distance < 10:
            return "far"
        else:
            return "very far"