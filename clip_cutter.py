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

def slice_candidate_clip(video_path, timestamp, clip_path, audio_path, half_duration=5.0):
    """
    Slice a centered 10s verification clip from video at the given timestamp.
    Also extracts the audio track as a mono WAV file for whistle detection.

    The clip is centered exactly on the Spivak proposal timestamp:
        [timestamp - half_duration, timestamp + half_duration]

    This is a TEMPORARY clip used only for audio verification — it should be
    deleted by the caller immediately after verify_event() returns.

    Args:
        video_path:    Path to the source MP4 video.
        timestamp:     Event timestamp in seconds (Spivak anchor point).
        clip_path:     Output path for the 10s MP4 clip.
        audio_path:    Output path for the extracted WAV audio.
        half_duration: Seconds before and after the anchor (default 5.0 → 10s total).

    Returns:
        (clip_path, audio_path_or_None):
            audio_path is None if the video has no audio stream or extraction fails.
    """
    ts = max(0.0, float(timestamp))
    start = max(0.0, ts - half_duration)
    duration = half_duration * 2  # 10 seconds total

    os.makedirs(os.path.dirname(clip_path) if os.path.dirname(clip_path) else ".", exist_ok=True)

    # Cut video clip — ultrafast preset since quality doesn't matter for verification
    cmd_clip = [
        FFMPEG_EXE, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-an",          # no audio in the clip — we extract it separately below
        clip_path
    ]
    result = subprocess.run(cmd_clip, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(clip_path):
        raise RuntimeError(f"FFmpeg error slicing candidate clip: {result.stderr.decode()[-400:]}")

    # Extract audio as mono 22050 Hz WAV — compatible with librosa
    cmd_audio = [
        FFMPEG_EXE, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # uncompressed PCM
        "-ar", "22050",             # sample rate matching librosa default
        "-ac", "1",                 # mono
        audio_path
    ]
    result_audio = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result_audio.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
        # Video may not have an audio stream — that's OK, verifier handles None gracefully
        return clip_path, None

    return clip_path, audio_path

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
