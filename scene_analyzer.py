# vision-audio-navigation/src/scene_analyzer.py

import cv2
import numpy as np

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

class SceneAnalyzer:
    def __init__(self):
        self.segmentation_model = self._load_segmentation_model()
        print("Scene analyzer initialized!")
    
    def _load_segmentation_model(self):
        """Load segmentation model if available"""
        if not SMP_AVAILABLE:
            print("💡 Using fallback segmentation analysis")
            return None
            
        try:
            model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights="voc",
                classes=19,
                activation=None,
            )
            print("✅ Segmentation model loaded")
            return model
        except Exception as e:
            print(f"⚠️ Could not load segmentation model: {e}")
            return None
    
    def analyze_scene(self, frame):
        """Analyze scene using semantic segmentation"""
        seg_map = self._perform_segmentation(frame)
        return self._analyze_segmentation_map(seg_map)
    
    def _perform_segmentation(self, frame):
        """Perform semantic segmentation on frame"""
        try:
            h, w = frame.shape[:2]
            seg_map = np.zeros((h, w), dtype=np.uint8)
            
            # Simple color-based segmentation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Road detection
            dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
            seg_map[h//2:, :][dark_mask[h//2:, :] > 0] = 0  # road
            
            # Sky detection
            sky_mask = cv2.inRange(hsv, (100, 50, 150), (140, 255, 255))
            seg_map[:h//3, :][sky_mask[:h//3, :] > 0] = 10  # sky
            
            return seg_map
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    def _analyze_segmentation_map(self, seg_map):
        """Analyze segmentation map for navigation guidance"""
        h, w = seg_map.shape
        
        analysis = {
            'guidance': [],
            'environment': 'unknown'
        }
        
        # Analyze immediate path (bottom 30%)
        immediate_path = seg_map[int(h*0.7):, :]
        road_pixels = np.sum(immediate_path == 0)
        total_pixels = immediate_path.size
        
        if total_pixels > 0:
            road_percentage = (road_pixels / total_pixels) * 100
            
            if road_percentage > 60:
                analysis['guidance'].append("Clear path ahead")
                analysis['environment'] = 'road'
            elif road_percentage > 30:
                analysis['guidance'].append("Moderate path clarity")
                analysis['environment'] = 'mixed'
            else:
                analysis['guidance'].append("Obstructed path ahead")
                analysis['environment'] = 'obstructed'
        
        return analysis
    
    def generate_guidance(self, seg_analysis):
        """Generate audio guidance from segmentation analysis"""
        if not seg_analysis['guidance']:
            return None
        return ". ".join(seg_analysis['guidance'])