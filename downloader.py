"""
YouTube Goal Compilation Splitter

This script downloads goal compilation videos from YouTube and automatically
splits them into individual goal clips using scene detection.

Requirements:
pip install yt-dlp opencv-python numpy scenedetect[opencv]

Usage:
1. Add YouTube URLs of goal compilations to the YOUTUBE_URLS list
2. Run the script
3. Individual goal clips will be saved in the output folder
"""

import os
import yt_dlp
import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import subprocess
import json

# Configuration
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=TjXZPJnzzbM",
]

OUTPUT_FOLDER = "training/skills"
TEMP_FOLDER = "temp_downloads"
MIN_CLIP_DURATION = 3   # Minimum seconds for a valid goal clip
MAX_CLIP_DURATION = 15  # Maximum seconds for a valid goal clip
SCENE_THRESHOLD = 30.0  # Scene detection sensitivity (lower = more cuts)

def main():
    print("Setting up directories...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    
    goal_count = 0
    
    for i, url in enumerate(YOUTUBE_URLS):
        print(f"\nProcessing video {i+1}/{len(YOUTUBE_URLS)}: {url}")
        
        # Download the video
        video_path = download_video(url, i+1)
        if not video_path:
            print(f"Failed to download video {i+1}")
            continue
        
        # Split into scenes (individual goals)
        scenes = detect_scenes(video_path)
        if not scenes:
            print(f"No scenes detected in video {i+1}")
            continue
        
        # Extract individual goal clips
        for j, (start_time, end_time) in enumerate(scenes):
            duration = end_time - start_time
            
            # Filter clips by duration
            if MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
                goal_count += 1
                output_path = os.path.join(OUTPUT_FOLDER, f"skills_{goal_count:03d}.mp4")
                
                success = extract_clip(video_path, start_time, end_time, output_path)
                if success:
                    print(f"Extracted goal {goal_count}: {duration:.1f}s")
                else:
                    goal_count -= 1  # Don't count failed extractions
        
        # Clean up downloaded video
        try:
            os.remove(video_path)
        except:
            pass
    
    print(f"\nCompleted! Extracted {goal_count} goal clips to {OUTPUT_FOLDER}")

def download_video(url, video_num):
    """Download video from YouTube using yt-dlp"""
    try:
        output_template = os.path.join(TEMP_FOLDER, f"video_{video_num}.%(ext)s")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Download 720p or lower to save space
            'outtmpl': output_template,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            print(f"Downloading: {video_title}")
            
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file (yt-dlp might use different extensions)
            for file in os.listdir(TEMP_FOLDER):
                if file.startswith(f"video_{video_num}."):
                    return os.path.join(TEMP_FOLDER, file)
            
            return None
            
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def detect_scenes(video_path):
    """Detect scene changes in the video (typically between different goals)"""
    try:
        # Create video manager and scene manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        
        # Add content detector with threshold
        scene_manager.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
        
        # Start video manager and detect scenes
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get list of scenes
        scene_list = scene_manager.get_scene_list()
        
        # Convert to list of (start_time, end_time) tuples in seconds
        scenes = []
        for scene in scene_list:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))
        
        print(f"Detected {len(scenes)} scenes")
        return scenes
        
    except Exception as e:
        print(f"Error detecting scenes: {str(e)}")
        return []

def extract_clip(input_path, start_time, end_time, output_path):
    """Extract a clip from the video using FFmpeg"""
    try:
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            output_path
        ]
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error extracting clip: {str(e)}")
        return False

def create_simple_splitter(video_path, num_splits=50):
    """
    Alternative simpler approach: split video into equal segments
    Use this if scene detection doesn't work well
    """
    try:
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        
        # Calculate segment duration
        segment_duration = duration / num_splits
        
        clips = []
        for i in range(num_splits):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            # Only include clips of reasonable duration
            if end_time - start_time >= MIN_CLIP_DURATION:
                clips.append((start_time, end_time))
        
        return clips
        
    except Exception as e:
        print(f"Error in simple splitter: {str(e)}")
        return []

if __name__ == "__main__":
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        exit(1)
    
    print("YouTube Goal Compilation Splitter")
    print("=" * 40)
    print("Make sure to:")
    print("1. Replace EXAMPLE URLs with real YouTube goal compilation URLs")
    print("2. Install required packages: pip install yt-dlp opencv-python scenedetect[opencv]")
    print("3. Have FFmpeg installed on your system")
    print()
    
    if all("EXAMPLE" in url for url in YOUTUBE_URLS):
        print("⚠️  Warning: Please replace the example URLs with real YouTube URLs!")
        print("Add goal compilation URLs like:")
        print('- "https://youtube.com/watch?v=dQw4w9WgXcQ"')
        print('- "https://youtube.com/watch?v=abc123def456"')
    else:
        main()