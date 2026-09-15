import os
import subprocess
import imageio_ffmpeg
import cv2

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

def get_video_info(video_path):
    """
    Get duration and FPS using OpenCV
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }

def cut_clip(input_video, start_time, duration, output_path):
    """
    Cut an MP4 video clip around a given time window using FFmpeg.
    Ensures web compatibility (H.264 + AAC + faststart).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_sec = max(0.0, float(start_time))
    dur_sec = max(1.0, float(duration))
    
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", input_video,
        "-t", f"{dur_sec:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "24",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"FFmpeg error cutting clip: {result.stderr[-400:]}")
    
    return output_path

def extract_thumbnail(input_video, timestamp, output_path):
    """
    Extract a single JPEG frame at the given timestamp for preview.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ts = max(0.0, float(timestamp))
    
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-ss", f"{ts:.3f}",
        "-i", input_video,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        # Fallback to OpenCV if ffmpeg fails on single frame
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)
        cap.release()
        
    return output_path
