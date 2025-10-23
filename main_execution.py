# vision-audio-navigation/src/main.py

import os
import sys
from .navigation_system import AudioNavigationSystem
from .utils import PlatformUtils

def main():
    print("=" * 60)
    print("ENHANCED AUDIO NAVIGATION SYSTEM")
    print("OBJECT-FIRST PRIORITY - PLATFORM AGNOSTIC")
    print("=" * 60)
    
    # Initialize system
    nav_system = AudioNavigationSystem()
    
    # Get file path from user
    file_path = input("\n📤 Enter path to image or video file: ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"✅ File found: {file_path}")
    
    # Process based on file type
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("\n🎥 Processing video...")
        output_video = nav_system.process_video(file_path)
        print(f"\n✅ Video processing complete: {output_video}")
        
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print("\n🖼️ Processing image...")
        output_image = nav_system.process_image(file_path)
        print(f"\n✅ Image processing complete: {output_image}")
        PlatformUtils.display_image(output_image)
        
    else:
        print("❌ Unsupported file format!")
    
    print("\n" + "=" * 60)
    print("✨ Processing Complete!")
    print("🎯 Object-First Navigation System")
    print("=" * 60)

if __name__ == "__main__":
    main()