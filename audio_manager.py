# vision-audio-navigation/src/audio_manager.py

import time
import threading
import os
from gtts import gTTS
from moviepy.editor import AudioFileClip, CompositeAudioClip
from .utils import PlatformUtils

class AudioManager:
    def __init__(self):
        self.use_audio = True
        self.audio_files = []
        self.audio_timestamps = []
        self.speaking = False
        self.audio_lock = threading.Lock()
        self.video_start_time = None
        print("Audio manager initialized!")
    
    def speak_text(self, text, timestamp=None):
        """Convert text to speech and play audio"""
        if not text or self.speaking:
            return
        
        with self.audio_lock:
            self.speaking = True
            try:
                if timestamp is None and self.video_start_time:
                    timestamp = time.time() - self.video_start_time
                timestamp = timestamp or 0
                
                timestamp_str = self._format_timestamp(timestamp)
                print(f"🔊 [{timestamp_str}] GUIDANCE: {text}")
                
                # Generate TTS audio
                audio_filename = self._generate_tts_audio(text, timestamp_str)
                
                # Store audio info
                self.audio_files.append(audio_filename)
                self.audio_timestamps.append({
                    'filename': audio_filename,
                    'timestamp': timestamp,
                    'timestamp_str': timestamp_str,
                    'text': text
                })
                
                # Play audio
                PlatformUtils.play_audio(audio_filename)
                
            except Exception as e:
                print(f"⚠️ Speech generation error: {e}")
            finally:
                self.speaking = False
                time.sleep(0.5)
    
    def _generate_tts_audio(self, text, timestamp_str):
        """Generate TTS audio file"""
        tts = gTTS(text=text, lang='en', slow=False)
        audio_filename = f"audio_{timestamp_str.replace(':', '-')}_{int(time.time() * 1000)}.mp3"
        tts.save(audio_filename)
        print(f"💾 Audio saved: {audio_filename}")
        return audio_filename
    
    def _format_timestamp(self, timestamp):
        """Format timestamp as MM:SS"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def merge_audio_into_video(self, video_path, output_path='final_with_audio.mp4'):
        """Merge all audio clips into video"""
        from moviepy.editor import VideoFileClip
        
        if not self.audio_timestamps:
            print("❌ No audio clips to merge")
            return video_path
        
        try:
            video = VideoFileClip(video_path)
            video_duration = video.duration
            
            # Create audio clips with timestamps
            audio_clips = []
            for audio_info in self.audio_timestamps:
                if os.path.exists(audio_info['filename']):
                    try:
                        audio_clip = AudioFileClip(audio_info['filename'])
                        audio_clip = audio_clip.set_start(audio_info['timestamp'])
                        audio_clips.append(audio_clip)
                    except Exception as e:
                        print(f"⚠️ Failed to load {audio_info['filename']}: {e}")
                        continue
            
            if not audio_clips:
                return video_path
            
            # Create composite audio and merge with video
            final_audio = CompositeAudioClip(audio_clips).set_duration(video_duration)
            final_video = video.set_audio(final_audio)
            final_video.write_videofile(output_path, verbose=False, logger=None)
            
            # Cleanup
            video.close()
            final_video.close()
            final_audio.close()
            for clip in audio_clips:
                clip.close()
            
            print(f"✅ Audio merged successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error merging audio: {e}")
            return video_path
    
    def generate_audio_report(self):
        """Generate transcript of all audio clips"""
        if not self.audio_timestamps:
            return None
        
        report_filename = 'audio_transcript.txt'
        with open(report_filename, 'w') as f:
            f.write("AUDIO NAVIGATION TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            
            for i, audio_info in enumerate(self.audio_timestamps, 1):
                f.write(f"[{audio_info['timestamp_str']}] Clip {i}\n")
                f.write(f"Text: {audio_info['text']}\n")
                f.write("-" * 50 + "\n")
        
        print(f"📄 Audio transcript saved: {report_filename}")
        return report_filename