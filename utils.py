# vision-audio-navigation/src/utils.py

import os
import sys
import pygame
from PIL import Image

class PlatformUtils:
    @staticmethod
    def is_colab():
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_jupyter():
        """Check if running in Jupyter notebook"""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except:
            return False
    
    @staticmethod
    def display_image(image_path):
        """Display image appropriately for the environment"""
        if PlatformUtils.is_jupyter() or PlatformUtils.is_colab():
            from IPython.display import display, Image
            display(Image(filename=image_path))
        else:
            img = Image.open(image_path)
            img.show()
    
    @staticmethod
    def play_audio(audio_path):
        """Play audio appropriately for the environment"""
        if PlatformUtils.is_jupyter() or PlatformUtils.is_colab():
            from IPython.display import Audio, display
            display(Audio(filename=audio_path))
        else:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Could not play audio: {e}")

class Config:
    """Configuration settings for the navigation system"""
    # Object detection
    YOLO_CONFIDENCE = 0.4
    YOLO_MODEL = 'yolov8n.pt'
    
    # Text detection
    TEXT_CONFIDENCE = 0.4
    MIN_TEXT_SIZE = 20
    
    # Audio settings
    ANNOUNCEMENT_COOLDOWN = 3
    MAX_ANNOUNCEMENTS = 4
    
    # Priority settings
    OBJECT_PRIORITY = {
        'vehicle': 10, 'person': 9, 'bicycle': 8, 'animal': 7,
        'chair': 6, 'bench': 6, 'traffic light': 5, 'stop sign': 5,
        'important_text': 4, 'text': 3, 'object': 2,
        'road': 1, 'sidewalk': 1, 'building': 1, 'vegetation': 1
    }