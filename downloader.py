"""
Soccer Compilation Splitter & Dataset Builder

Downloads soccer compilation videos (or processes local video files)
and automatically splits them into individual clips using scene detection
or fixed-interval slicing for ML training.

Usage:
  # Split YouTube goal compilation into training/goals/
  python downloader.py --url "https://www.youtube.com/watch?v=EXAMPLE" --label goals

  # Split local video file with scene cuts
  python downloader.py --input /path/to/video.mp4 --label goals

  # Split full match into 8-second regular clips for negative samples (no_events)
  python downloader.py --input /path/to/match.mp4 --label no_events --split-interval 8
"""

import os
import sys
import argparse
import subprocess
import cv2
import yt_dlp
from scenedetect import detect, ContentDetector

def download_video(url, temp_folder="temp_downloads", video_num=1):
    """Download video from YouTube using yt-dlp"""
    try:
        os.makedirs(temp_folder, exist_ok=True)
        output_template = os.path.join(temp_folder, f"video_{video_num}.%(ext)s")

        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': output_template,
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            print(f"[Downloader] Downloading: {video_title}...")
            ydl.download([url])

            for file in os.listdir(temp_folder):
                if file.startswith(f"video_{video_num}."):
                    return os.path.join(temp_folder, file)

        return None
    except Exception as e:
        print(f"[Downloader] Error downloading video: {e}")
        return None

def detect_scenes(video_path, threshold=30.0):
    """Detect scene cuts using modern PySceneDetect"""
    try:
        print(f"[SceneDetect] Analyzing cuts in {video_path} (threshold={threshold})...")
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        scenes = []
        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            scenes.append((start_sec, end_sec))
        print(f"[SceneDetect] Detected {len(scenes)} scenes.")
        return scenes
    except Exception as e:
        print(f"[SceneDetect] Error detecting scenes: {e}")
        return []

def uniform_split(video_path, interval=8.0, min_duration=3.0):
    """Split video into fixed intervals (ideal for continuous play / no_events)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Splitter] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()

    scenes = []
    curr = 0.0
    while curr < duration:
        end = min(curr + interval, duration)
        if end - curr >= min_duration:
            scenes.append((curr, end))
        curr += interval

    print(f"[Splitter] Generated {len(scenes)} uniform segments ({interval}s each).")
    return scenes

def extract_clip(input_path, start_time, duration, output_path):
    """Fast clip extraction using FFmpeg copy stream"""
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"[FFmpeg] Error extracting clip: {e}")
        return False

def process_video(video_path, output_dir, label, threshold=30.0, min_dur=3, max_dur=15, split_interval=None):
    os.makedirs(output_dir, exist_ok=True)

    if split_interval and split_interval > 0:
        scenes = uniform_split(video_path, interval=split_interval, min_duration=min_dur)
    else:
        scenes = detect_scenes(video_path, threshold=threshold)
        if not scenes:
            print("[Splitter] Scene detection found 0 cuts, falling back to 8s uniform split...")
            scenes = uniform_split(video_path, interval=8.0, min_duration=min_dur)

    # Find existing max index in output_dir to avoid overwriting
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"{label}_") and f.endswith(".mp4")]
    start_idx = len(existing_files) + 1

    extracted_count = 0
    for start_time, end_time in scenes:
        dur = end_time - start_time
        if split_interval or (min_dur <= dur <= max_dur):
            output_path = os.path.join(output_dir, f"{label}_{start_idx:04d}.mp4")
            success = extract_clip(video_path, start_time, dur, output_path)
            if success:
                start_idx += 1
                extracted_count += 1
                print(f"  [OK] Saved clip {label}_{start_idx-1:04d}.mp4 ({dur:.1f}s)")

    print(f"[Splitter] Done! Extracted {extracted_count} clips to {output_dir}")
    return extracted_count

def main():
    parser = argparse.ArgumentParser(description="Soccer Video Compilation Splitter & Dataset Builder")
    parser.add_argument("--url", type=str, default=None, help="YouTube video URL to download & split")
    parser.add_argument("--input", type=str, default=None, help="Local video file path to split")
    parser.add_argument("--label", type=str, default="goals", choices=["goals", "attacks", "no_events", "skills"],
                        help="Event class label (goals, attacks, no_events)")
    parser.add_argument("--output-dir", type=str, default="./training", help="Base directory for training dataset")
    parser.add_argument("--threshold", type=float, default=30.0, help="PySceneDetect cut sensitivity (default: 30.0)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Min clip length in seconds")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Max clip length in seconds")
    parser.add_argument("--split-interval", type=float, default=None,
                        help="If specified, splits video into uniform intervals of N seconds (great for no_events)")

    args = parser.parse_args()

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except Exception:
        print("Error: ffmpeg is required but not found in PATH.")
        sys.exit(1)

    target_dir = os.path.join(args.output_dir, args.label)

    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input video '{args.input}' not found.")
            sys.exit(1)
        process_video(args.input, target_dir, args.label, args.threshold,
                      args.min_duration, args.max_duration, args.split_interval)

    elif args.url:
        temp_dir = "temp_downloads"
        vpath = download_video(args.url, temp_dir, 1)
        if vpath:
            process_video(vpath, target_dir, args.label, args.threshold,
                          args.min_duration, args.max_duration, args.split_interval)
            try:
                os.remove(vpath)
            except Exception:
                pass
        else:
            print("Failed to download video from URL.")

    else:
        print("Usage: python downloader.py --url <YOUTUBE_URL> --label goals")
        print("   or: python downloader.py --input <VIDEO_FILE> --label no_events --split-interval 8")
        parser.print_help()

if __name__ == "__main__":
    main()