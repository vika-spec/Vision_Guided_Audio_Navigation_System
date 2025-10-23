# vision-audio-navigation/src/text_detector.py

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from .utils import Config

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class TextDetector:
    def __init__(self):
        self.reader = self._initialize_reader()
        self.important_keywords = [
            'exit', 'entrance', 'warning', 'danger', 'caution', 'stop',
            'stairs', 'elevator', 'escalator', 'crosswalk', 'curb',
            'emergency', 'hospital', 'police', 'fire', 'help'
        ]
        self.text_size_reference = 100
        print("Text detector initialized!")
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader if available"""
        if EASYOCR_AVAILABLE:
            try:
                return easyocr.Reader(['en'])
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
        print("💡 EasyOCR not available. Text detection disabled.")
        return None
    
    def detect_text(self, frame, frame_width):
        """Detect and process text in frame"""
        if self.reader is None:
            return []
        
        try:
            processed_frame = self._preprocess_image(frame)
            results = self.reader.readtext(
                processed_frame,
                decoder='beamsearch',
                beamWidth=5,
                text_threshold=0.3,
                link_threshold=0.3
            )
            
            detected_texts = []
            for (bbox, text, confidence) in results:
                if confidence > Config.TEXT_CONFIDENCE and len(text.strip()) > 1:
                    text_data = self._process_text_detection(bbox, text, confidence, frame_width)
                    if text_data:
                        detected_texts.append(text_data)
            
            return detected_texts
            
        except Exception as e:
            print(f"Text detection error: {e}")
            return []
    
    def _preprocess_image(self, image):
        """Preprocess image to enhance text detection"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _process_text_detection(self, bbox, text, confidence, frame_width):
        """Process individual text detection"""
        clean_text = text.strip().lower()
        
        if len(bbox) >= 4:
            y_coords = [point[1] for point in bbox]
            text_height = max(y_coords) - min(y_coords)
            distance = self._calculate_text_distance(text_height)
            distance_category = self._get_distance_category(distance)
            is_important = any(keyword in clean_text for keyword in self.important_keywords)
            
            return {
                'type': 'text',
                'text': clean_text,
                'confidence': confidence,
                'bbox': bbox,
                'position': self._get_text_position(bbox, frame_width),
                'distance': distance,
                'distance_category': distance_category,
                'is_important': is_important,
                'priority': 4 if is_important else 3
            }
        return None
    
    def _calculate_text_distance(self, bbox_height):
        """Estimate distance to text"""
        if bbox_height <= 0:
            return 10.0
        distance = (self.text_size_reference * 2.0) / bbox_height
        return max(0.5, min(distance, 15.0))
    
    def _get_text_position(self, bbox, frame_width):
        """Determine text position in frame"""
        x_coords = [point[0] for point in bbox]
        x_center = sum(x_coords) / len(x_coords)
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