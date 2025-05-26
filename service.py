import cv2
import numpy as np
import os
import time
from collections import deque
from mongo import save_events_to_db
from ml_model import SoccerEventDetector, download_pretrained_model

EVENT_DETECTOR = None

def initialize_model():
    """
    Initialize the ML model if not already loaded
    """
    global EVENT_DETECTOR
    
    if EVENT_DETECTOR is None:
        model_path = "models/soccer_events_model"
        
        if os.path.exists(model_path):
            EVENT_DETECTOR = SoccerEventDetector(model_path)
        else:
            print("WARNING: No trained model found. Using fallback detection methods.")
            EVENT_DETECTOR = SoccerEventDetector()
    
    return EVENT_DETECTOR

def extract_soccer_events(video_path):
    """
    Extract soccer events (Goals and Attacks) from a video file using ML
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        list: List of event dictionaries with timestamp and event type
    """
    # Initialize ML model
    model = initialize_model()
    events = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # We'll use a sliding window approach to analyze video segments
    sequence_length = model.sequence_length
    frame_buffer = deque(maxlen=sequence_length)
    
    # Skip frames to speed up processing - analyze every n frames
    frame_skip = 3  # Analyze every 3rd frame
    
    # Cooldown period to avoid detecting the same event multiple times
    event_cooldown = {"Goal": 0, "Attack": 0}
    cooldown_frames = {"Goal": int(10 * fps), "Attack": int(5 * fps)}
    
    # Process frames
    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process every nth frame to improve performance
        if frame_idx % frame_skip == 0:
            frame_buffer.append(frame.copy())
            
            # When buffer is full, run prediction
            if len(frame_buffer) == sequence_length:
                # Calculate timestamp in seconds
                timestamp = frame_idx / fps
                
                # Run ML prediction
                predicted_class_idx, confidence = model.predict_sequence(list(frame_buffer))
                predicted_class = model.classes[predicted_class_idx]
                
                # If an event is detected with high confidence
                if predicted_class != 'No Event' and confidence > 0.71:
                    # Check cooldown period
                    if event_cooldown[predicted_class] <= 0:
                        events.append({
                            "timestamp": timestamp,
                            "event_type": predicted_class,
                            "confidence": float(confidence)
                        })
                        
                        # Set cooldown period
                        event_cooldown[predicted_class] = cooldown_frames[predicted_class]
        
        # Update cooldown counters
        for event_type in event_cooldown:
            if event_cooldown[event_type] > 0:
                event_cooldown[event_type] -= 1
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    
    # Sort events by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    # Save events to MongoDB
    video_filename = os.path.basename(video_path)
    save_events_to_db(video_filename, events)
    
    events.sort(key=lambda x: x['timestamp'])

    # Merge nearby events
    events = merge_nearby_events(events, time_threshold=3.0)
    
    return events

def extract_frames_around_timestamp(video_path, timestamp, window_seconds=5):
    """
    Extract frames around a specific timestamp for verification
    
    Args:
        video_path (str): Path to the video file
        timestamp (float): Timestamp in seconds
        window_seconds (int): Number of seconds to extract before and after
        
    Returns:
        list: List of paths to extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate start and end frames
    start_frame = max(0, int((timestamp - window_seconds) * fps))
    end_frame = int((timestamp + window_seconds) * fps)
    
    # Create output directory
    video_filename = os.path.basename(video_path).split('.')[0]
    output_dir = f"./static/frames/{video_filename}/{int(timestamp)}"
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    
    # Set frame position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save every 5th frame to reduce number of images
        if current_frame % 5 == 0:
            frame_path = f"{output_dir}/frame_{current_frame}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        current_frame += 1
    
    cap.release()
    
    return frame_paths

# Add this function to service.py
def merge_nearby_events(events, time_threshold=3.0):
    """
    Merge events of the same type that occur within a specified time threshold
    """
    if not events:
        return []
        
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x['timestamp'])
    
    merged_events = [sorted_events[0]]
    
    for event in sorted_events[1:]:
        last_event = merged_events[-1]
        
        # If same event type and close in time, merge by keeping the one with higher confidence
        if (event['event_type'] == last_event['event_type'] and 
            abs(event['timestamp'] - last_event['timestamp']) <= time_threshold):
            
            # Keep the event with higher confidence
            if event['confidence'] > last_event['confidence']:
                merged_events[-1] = event
        else:
            merged_events.append(event)
            
    return merged_events