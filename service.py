import os
import cv2
import time
import shutil
from collections import deque
from ml_model import SoccerEventDetector
from clip_cutter import cut_clip, extract_thumbnail, get_video_info, slice_candidate_clip
from minio_storage import upload_file
from gdrive_downloader import download_gdrive_video
from spivak_service import extract_soccer_events_spivak, SpivakSoccerSpotter

try:
    from verifier import verify_event as _verify_event
    _VERIFIER_AVAILABLE = True
except ImportError:
    _VERIFIER_AVAILABLE = False
    print("[service] Warning: verifier module unavailable — install librosa/scipy to enable whistle verification.")

def verify_event(audio_wav_path, event_type, spivak_confidence, mode="full"):
    """Thin wrapper — falls back to accepting all events if verifier deps not installed."""
    if not _VERIFIER_AVAILABLE or mode == "disabled":
        return True
    return _verify_event(audio_wav_path, event_type, spivak_confidence, mode)

EVENT_DETECTOR = None

def initialize_model():
    """
    Initialize the ML model if not already loaded
    """
    global EVENT_DETECTOR
    if EVENT_DETECTOR is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "soccer_events_model")
        EVENT_DETECTOR = SoccerEventDetector(model_path)
    return EVENT_DETECTOR

def merge_nearby_events(events, time_threshold=4.0):
    """
    Merge events of the same type that occur within a specified time threshold
    """
    if not events:
        return []
        
    sorted_events = sorted(events, key=lambda x: x['timestamp'])
    merged = [sorted_events[0]]
    
    for event in sorted_events[1:]:
        last_event = merged[-1]
        
        # If same event type and close in time, keep the one with higher confidence
        if (event['event_type'] == last_event['event_type'] and 
            abs(event['timestamp'] - last_event['timestamp']) <= time_threshold):
            if event['confidence'] > last_event['confidence']:
                merged[-1] = event
        else:
            merged.append(event)
            
    return merged

def extract_soccer_events(video_path, min_confidence=0.65, frame_skip=3, progress_callback=None):
    """
    Extract soccer events (Goal and Attack) using ML sliding window
    """
    model = initialize_model()
    events = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka file video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    
    sequence_length = model.sequence_length
    frame_buffer = deque(maxlen=sequence_length)
    
    cooldown_frames = {"Goal": int(10 * fps), "Attack": int(5 * fps)}
    event_cooldown = {"Goal": 0, "Attack": 0}
    
    frame_idx = 0
    last_reported_pct = 0
    
    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_skip == 0:
            frame_buffer.append(frame.copy())
            
            if len(frame_buffer) == sequence_length:
                timestamp = frame_idx / fps
                pred_idx, confidence, class_name = model.predict_sequence(list(frame_buffer))
                
                if class_name in ['Goal', 'Attack'] and confidence >= min_confidence:
                    if event_cooldown[class_name] <= 0:
                        events.append({
                            "timestamp": round(timestamp, 2),
                            "minute": int(timestamp // 60),
                            "event_type": class_name.upper(),
                            "confidence": round(confidence, 3)
                        })
                        event_cooldown[class_name] = cooldown_frames[class_name]
                        
        for k in event_cooldown:
            if event_cooldown[k] > 0:
                event_cooldown[k] -= 1
                
        # Report progress periodically
        if frame_idx % (frame_skip * 30) == 0 and progress_callback and frame_count > 0:
            pct = int((frame_idx / frame_count) * 100)
            if pct > last_reported_pct:
                last_reported_pct = pct
                progress_callback(pct, f"Menganalisis frame video... ({pct}%)")
                
        frame_idx += 1
        
    cap.release()
    
    # Merge nearby events
    events = merge_nearby_events(events, time_threshold=4.0)
    return events, duration

def process_match_video_pipeline(job_id, gdrive_url_or_path, event_id, match_id="general", options=None, progress_callback=None):
    """
    Full pipeline:
    1. Download GDrive video
    2. Extract events
    3. Cut clips & extract thumbnails
    4. Upload to MinIO
    5. Cleanup local scratch files
    """
    options = options or {}
    engine = options.get("engine", "spivak").lower()
    default_conf = 0.25 if engine == "spivak" else 0.65
    min_confidence = float(options.get("min_confidence", default_conf))
    clip_padding_before = float(options.get("clip_padding_before", 6.0))
    clip_padding_after = float(options.get("clip_padding_after", 6.0))
    clip_duration = clip_padding_before + clip_padding_after
    
    scratch_dir = os.path.join(os.path.dirname(__file__), "scratch", job_id)
    os.makedirs(scratch_dir, exist_ok=True)
    
    raw_video_path = None
    try:
        # Step 1: Download or locate video
        if os.path.exists(gdrive_url_or_path):
            raw_video_path = gdrive_url_or_path
        else:
            raw_video_path = os.path.join(scratch_dir, "raw_video.mp4")
            if progress_callback:
                progress_callback("DOWNLOADING", 5, "Memulai unduhan video dari Google Drive...")
                
            def dl_hook(pct, msg):
                if progress_callback:
                    progress_callback("DOWNLOADING", pct, msg)
                    
            download_gdrive_video(gdrive_url_or_path, raw_video_path, progress_callback=dl_hook)
            
        # Step 2: AI Event Analysis
        if progress_callback:
            engine_name = "Spivak SoccerNet" if engine == "spivak" else "Native EfficientNet+LSTM"
            progress_callback("ANALYZING", 0, f"Memulai analisis visual AI ({engine_name})...")
            
        def ai_hook(pct, msg):
            if progress_callback:
                progress_callback("ANALYZING", pct, msg)
                
        if engine == "spivak":
            events, duration = extract_soccer_events_spivak(
                raw_video_path,
                min_confidence=min_confidence,
                progress_callback=ai_hook
            )
        else:
            events, duration = extract_soccer_events(
                raw_video_path,
                min_confidence=min_confidence,
                frame_skip=3,
                progress_callback=ai_hook
            )
        
        print(f"[{job_id}] [{engine}] Found {len(events)} events in video duration {duration:.1f}s (min_confidence={min_confidence})", flush=True)
        if len(events) == 0:
            print(f"[{job_id}] [{engine}] Notice: 0 events detected with threshold {min_confidence}. Try lowering threshold or switching engine.", flush=True)
        
        # Step 2b: Verify Spivak candidates via secondary audio signal (whistle)
        # Native engine events skip this step — they use a different confidence model.
        verification_mode = options.get("verification", "full")
        if engine == "spivak" and verification_mode != "disabled" and len(events) > 0:
            if progress_callback:
                progress_callback("ANALYZING", 97, f"Memverifikasi {len(events)} kandidat momen...")
            
            verified_events = []
            for ev in events:
                ts = ev["timestamp"]
                cand_clip = os.path.join(scratch_dir, f"cand_{int(ts)}.mp4")
                cand_audio = os.path.join(scratch_dir, f"cand_{int(ts)}.wav")
                actual_audio = None
                
                try:
                    _, actual_audio = slice_candidate_clip(
                        raw_video_path, ts, cand_clip, cand_audio
                    )
                except Exception as slice_err:
                    print(f"[{job_id}] Warning: candidate clip slice failed at {ts}s: {slice_err}", flush=True)
                
                accepted = verify_event(
                    audio_wav_path=actual_audio,
                    event_type=ev["event_type"],
                    spivak_confidence=ev["confidence"],
                    mode=verification_mode
                )
                
                # Always clean up temp files immediately to preserve disk space
                for tmp in [cand_clip, cand_audio]:
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
                
                if accepted:
                    verified_events.append(ev)
                else:
                    print(f"[{job_id}] Rejected by verifier: {ev['event_type']} at {ts}s conf={ev['confidence']}", flush=True)
            
            print(f"[{job_id}] Verification: {len(events)} candidates → {len(verified_events)} accepted", flush=True)
            events = verified_events
        
        # Step 3: Cut final highlight clips & Upload to MinIO
        if progress_callback:
            progress_callback("UPLOADING", 0, f"Memotong {len(events)} klip highlight dan mengunggah ke MinIO...")
            
        highlights = []
        clean_match_id = match_id or "general"
        clean_event_id = event_id or "general"
        
        for idx, ev in enumerate(events):
            ts = ev["timestamp"]
            ev_type = ev["event_type"].lower()
            start_time = max(0.0, ts - clip_padding_before)
            
            local_clip = os.path.join(scratch_dir, f"clip_{int(ts)}_{ev_type}.mp4")
            local_thumb = os.path.join(scratch_dir, f"thumb_{int(ts)}.jpg")
            
            # Cut clip
            cut_clip(raw_video_path, start_time, clip_duration, local_clip)
            
            # Extract thumb
            extract_thumbnail(raw_video_path, ts, local_thumb)
            
            # MinIO paths
            minio_clip_key = f"highlights/{clean_event_id}/{clean_match_id}/clip_{int(ts)}_{ev_type}.mp4"
            minio_thumb_key = f"highlights/{clean_event_id}/{clean_match_id}/thumb_{int(ts)}.jpg"
            
            # Upload
            upload_file(local_clip, minio_clip_key, content_type="video/mp4")
            upload_file(local_thumb, minio_thumb_key, content_type="image/jpeg")
            
            highlights.append({
                "timestamp": ts,
                "minute": ev["minute"],
                "event_type": ev["event_type"],
                "confidence": ev["confidence"],
                "minio_clip_path": minio_clip_key,
                "minio_thumb_path": minio_thumb_key,
                "clip_duration": clip_duration
            })
            
            if progress_callback:
                pct = int(((idx + 1) / max(1, len(events))) * 100)
                progress_callback("UPLOADING", pct, f"Mengunggah highlight {idx + 1}/{len(events)} ke MinIO...")

        return {
            "duration": duration,
            "highlights": highlights
        }
        
    finally:
        # Step 4: Cleanup scratch folder to save disk
        try:
            if os.path.exists(scratch_dir):
                shutil.rmtree(scratch_dir, ignore_errors=True)
            if raw_video_path and os.path.exists(raw_video_path) and "scratch" in os.path.abspath(raw_video_path):
                os.remove(raw_video_path)
                parent = os.path.dirname(raw_video_path)
                if os.path.exists(parent) and "uploads" in parent and not os.listdir(parent):
                    os.rmdir(parent)
        except Exception as e:
            print(f"[{job_id}] Warning: cleanup failed for {scratch_dir}: {e}")