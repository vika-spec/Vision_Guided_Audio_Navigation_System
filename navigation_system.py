# vision-audio-navigation/src/navigation_system.py

import cv2
import time
import threading
from collections import deque

from .object_detector import ObjectDetector
from .text_detector import TextDetector
from .scene_analyzer import SceneAnalyzer
from .audio_manager import AudioManager
from .video_processor import VideoProcessor
from .utils import Config

class AudioNavigationSystem:
    def __init__(self):
        # Initialize components
        self.object_detector = ObjectDetector()
        self.text_detector = TextDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.audio_manager = AudioManager()
        self.video_processor = VideoProcessor()
        
        # State management
        self.frame_width = 0
        self.frame_height = 0
        self.last_announcement = time.time()
        self.detected_items = set()
        
        print("🎯 Audio Navigation System Initialized!")
    
    def process_frame(self, frame):
        """Process single frame with object-first priority"""
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Run all detection modalities
        objects_info = self.object_detector.detect_objects(frame, self.frame_width)
        text_info = self._detect_text_with_cooldown()
        seg_analysis = self.scene_analyzer.analyze_scene(frame)
        
        # Combine all detections
        all_detections = objects_info + text_info
        
        # Generate visualization
        frame = self._visualize_detections(frame, objects_info, text_info)
        
        # Generate and speak announcements
        message = self._generate_announcement(all_detections, seg_analysis)
        if message and message != "Path clear":
            threading.Thread(target=self.audio_manager.speak_text, args=(message,)).start()
            self.last_announcement = time.time()
        
        return frame, message, len(objects_info), len(text_info)
    
    def _detect_text_with_cooldown(self):
        """Detect text with cooldown to avoid over-processing"""
        current_time = time.time()
        if (current_time - self.last_announcement) > 2.0:
            # Text detection logic with filtering
            return self.text_detector.detect_text(self._current_frame, self.frame_width)
        return []
    
    def _visualize_detections(self, frame, objects_info, text_info):
        """Add visualization overlays to frame"""
        # Visualize objects
        for obj in objects_info:
            frame = self._draw_object_annotation(frame, obj)
        
        # Visualize text (less prominent)
        for text in text_info:
            frame = self._draw_text_annotation(frame, text)
        
        # Add status overlay
        frame = self._add_status_overlay(frame, len(objects_info), len(text_info))
        
        return frame
    
    def _draw_object_annotation(self, frame, obj):
        """Draw object bounding box and label"""
        x1, y1, x2, y2 = obj['bbox']
        label = obj['label']
        
        # Color coding based on object type
        color_map = {
            'vehicle': (0, 0, 255),      # Red
            'person': (0, 255, 255),     # Yellow
            'bicycle': (255, 0, 0),      # Blue
            'animal': (255, 165, 0),     # Orange
        }
        color = color_map.get(label, (0, 255, 0))  # Green for others
        
        # Draw bounding box
        thickness = 3 if label in ['vehicle', 'person', 'bicycle'] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label_text = f"{label.upper()} {obj['distance_category']}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, label_text, (x1+5, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _draw_text_annotation(self, frame, text_data):
        """Draw text bounding box and label"""
        bbox = text_data['bbox']
        text = text_data['text']
        is_important = text_data['is_important']
        
        # Subdued colors for text
        color = (200, 0, 200) if is_important else (200, 200, 0)
        thickness = 2 if is_important else 1
        
        # Draw polygon around text
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)
        
        return frame
    
    def _add_status_overlay(self, frame, obj_count, text_count):
        """Add status information overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        status_text = f"Objects: {obj_count} | Texts: {text_count}"
        cv2.putText(frame, status_text, (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _generate_announcement(self, all_detections, seg_analysis):
        """Generate navigation announcement with object-first priority"""
        if not all_detections:
            return "Path clear"
        
        # Sort by priority (objects first)
        all_detections.sort(key=self._get_comprehensive_priority, reverse=True)
        
        messages = []
        announced_count = 0
        
        for item in all_detections:
            if announced_count >= Config.MAX_ANNOUNCEMENTS:
                break
            
            if item['type'] == 'object':
                # Object announcements (highest priority)
                message = self._format_object_announcement(item)
                messages.append(message)
                announced_count += 1
            elif item['type'] == 'text' and announced_count < Config.MAX_ANNOUNCEMENTS:
                # Text announcements (lower priority, only if space)
                message = self._format_text_announcement(item)
                messages.append(message)
                announced_count += 1
        
        return ". ".join(messages) if messages else "Path clear"
    
    def _get_comprehensive_priority(self, item):
        """Calculate comprehensive priority score"""
        base_priority = item.get('priority', 1)
        distance = item.get('distance', 10)
        position = item.get('position', 'right')
        
        # Distance factor (closer = higher priority)
        distance_factor = max(0, 10 - distance) / 2
        
        # Position factor (center = higher priority)
        position_factor = 2 if position == 'center' else 1
        
        # Critical objects in center get boost
        if (item.get('type') == 'object' and 
            position == 'center' and 
            distance < 3 and 
            item.get('label') in ['vehicle', 'person', 'bicycle']):
            distance_factor += 3
        
        return base_priority * position_factor + distance_factor
    
    def _format_object_announcement(self, item):
        """Format object detection into announcement"""
        label = item['label']
        position = item['position']
        distance_category = item['distance_category']
        distance = item['distance']
        
        if position == "center" and distance < 3:
            if label in ['vehicle', 'person', 'bicycle']:
                return f"⚠️ WARNING! {label} directly ahead, {distance_category}"
            else:
                return f"Obstacle directly ahead, {distance_category}"
        else:
            return f"{label} on your {position}, {distance_category}"
    
    def _format_text_announcement(self, item):
        """Format text detection into announcement"""
        text = item['text']
        position = item['position']
        distance_category = item['distance_category']
        
        if item['is_important']:
            return f"Sign: {text} {distance_category} on your {position}"
        else:
            return f"Text: {text}"
    
    # Public interface methods
    def process_video(self, video_path, output_path='output_navigation.mp4'):
        """Process video file"""
        self.audio_manager.video_start_time = time.time()
        
        def process_frame_callback(frame):
            processed_frame, _, _, _ = self.process_frame(frame)
            return processed_frame
        
        output_video = self.video_processor.process_video(
            video_path, output_path, process_frame_callback
        )
        
        # Merge audio if available
        if self.audio_manager.audio_timestamps:
            final_output = 'final_with_audio.mp4'
            return self.audio_manager.merge_audio_into_video(output_video, final_output)
        
        return output_video
    
    def process_image(self, image_path):
        """Process single image"""
        def process_frame_callback(frame):
            processed_frame, message, obj_count, text_count = self.process_frame(frame)
            print(f"Detected {obj_count} objects and {text_count} text elements")
            if message:
                print(f"Navigation: {message}")
            return processed_frame
        
        return self.video_processor.process_image(image_path, process_frame_callback)