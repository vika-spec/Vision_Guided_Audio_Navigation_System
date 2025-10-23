# vision-audio-navigation/src/video_processor.py

import cv2
import os

class VideoProcessor:
    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0
    
    def process_video(self, video_path, output_path, process_frame_callback):
        """Process video file frame by frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame using callback
                processed_frame = process_frame_callback(frame)
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                    
        finally:
            cap.release()
            out.release()
        
        print(f"✅ Video processing complete: {output_path}")
        return output_path
    
    def process_image(self, image_path, process_frame_callback):
        """Process single image"""
        frame = cv2.imread(image_path)
        processed_frame = process_frame_callback(frame)
        
        output_path = 'output_navigation.jpg'
        cv2.imwrite(output_path, processed_frame)
        print(f"✅ Image processing complete: {output_path}")
        return output_path