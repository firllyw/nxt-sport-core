from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import cv2
import datetime
import uuid
import threading
import time
from werkzeug.utils import secure_filename
from service import extract_soccer_events, extract_frames_around_timestamp
from mongo import get_all_videos_with_events, save_video_metadata, get_video_by_string_id
import json

api_blueprint = Blueprint('api', __name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    """Check if a filename has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_blueprint.route('/upload-and-extract', methods=['POST'])
def upload_and_extract():
    """
    Upload a video file with metadata and extract events in the background
    """
    try:
        # Check if request has the file part
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        
        # Check if file was selected
        if video_file.filename == '':
            return jsonify({"error": "No video selected"}), 400
        
        # Check file type
        if not allowed_file(video_file.filename):
            return jsonify({"error": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Get metadata from form
        event_name = request.form.get('event_name', 'Unnamed Event')
        event_date = request.form.get('event_date', datetime.datetime.now().strftime('%Y-%m-%d'))
        event_time = request.form.get('event_time', datetime.datetime.now().strftime('%H:%M'))
        event_place = request.form.get('event_place', 'Unknown Location')
        
        # Generate unique ID for this video
        video_id = str(uuid.uuid4())
        
        # Get file extension
        _, extension = os.path.splitext(video_file.filename)
        
        # Create unique filename
        unique_filename = f"{video_id}{extension}"
        
        # Ensure video directory exists
        video_dir = os.path.join(current_app.root_path, 'video')
        os.makedirs(video_dir, exist_ok=True)
        
        # Save file to video directory
        video_path = os.path.join(video_dir, unique_filename)
        video_file.save(video_path)
        
        # Create metadata
        video_metadata = {
            "video_id": video_id,
            "video_filename": unique_filename,
            "original_filename": secure_filename(video_file.filename),
            "event_name": event_name,
            "event_date": event_date,
            "event_time": event_time,
            "event_place": event_place,
            "upload_date": datetime.datetime.now().isoformat(),
            "processing_status": "processing"
        }
        
        # Save metadata to MongoDB
        save_video_metadata(video_metadata)
        
        # Start background processing
        thread = threading.Thread(target=process_video_in_background, 
                                 args=(video_path, unique_filename, video_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Video uploaded successfully and processing started",
            "video_id": video_id,
            "video_filename": unique_filename
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_video_in_background(video_path, filename, video_id):
    """
    Process the video in a background thread
    """
    try:
        print(f"Starting background processing for {filename}")
        
        # Update status to processing
        update_processing_status(video_id, "processing")
        
        # Extract events using ML model
        events = extract_soccer_events(video_path)
        
        # Calculate video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Update MongoDB with events and status
        update_video_with_events(video_id, events, duration)
        
        # Update status to completed
        update_processing_status(video_id, "completed")
        
        print(f"Background processing completed for {filename}")
    
    except Exception as e:
        print(f"Error in background processing: {e}")
        update_processing_status(video_id, "failed", str(e))

def update_processing_status(video_id, status, error=None):
    """
    Update the processing status in MongoDB
    """
    from mongo import update_video_status
    update_video_status(video_id, status, error)

def update_video_with_events(video_id, events, duration):
    """
    Update the video document with extracted events and duration
    """
    from mongo import update_video_events
    update_video_events(video_id, events, duration)

@api_blueprint.route('/processing-status/<video_id>', methods=['GET'])
def get_processing_status(video_id):
    """
    Get the processing status of a video
    """
    try:
        from mongo import get_video_processing_status
        status = get_video_processing_status(video_id)
        
        if status is None:
            return jsonify({"error": "Video not found"}), 404
            
        return jsonify(status), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@api_blueprint.route('/extract-events', methods=['POST'])
def extract_events():
    # Get filename from request parameters
    filename = request.args.get('filename')
    
    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400
    
    # Check if file exists in video folder
    video_path = os.path.join('video', filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"File {filename} not found in video folder"}), 404
    
    try:
        # Call service to extract events
        event_data = extract_soccer_events(video_path)
        
        return jsonify({
            "status": "success",
            "message": "Events extracted successfully",
            "events": event_data
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/videos', methods=['GET'])
def get_videos():
    """
    Get all videos with events from MongoDB
    """
    try:
        # Get videos from MongoDB that have events
        videos_data = get_all_videos_with_events()
        
        # Format the response for the frontend
        videos = []
        for video in videos_data:
            # Calculate total events
            total_events = video.get('goal_count', 0) + video.get('attack_count', 0)
            
            # Format the date
            processed_date = video.get('processed_at')
            if processed_date:
                if isinstance(processed_date, str):
                    # Parse ISO format date
                    try:
                        date_obj = datetime.datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%b %d, %Y')
                    except:
                        formatted_date = processed_date
                else:
                    # Already a datetime object
                    formatted_date = processed_date.strftime('%b %d, %Y')
            else:
                formatted_date = "Unknown date"
            
            videos.append({
                "id": str(video.get('_id')),
                "filename": video.get('video_filename'),
                "event_name": video.get('event_name'),
                "event_date": video.get('event_date'),
                "event_place": video.get('event_place'),
                "title": os.path.splitext(video.get('video_filename', ''))[0].replace("_", " "),
                "events_count": total_events,
                "goal_count": video.get('goal_count', 0),
                "attack_count": video.get('attack_count', 0),
                "date": formatted_date
            })
        
        return jsonify({
            "videos": videos
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/videos/<video_id>', methods=['GET'])
def get_video_details(video_id):
    """
    Get details of a specific video and its events by MongoDB ID
    """
    try:
        # Get video data using direct string comparison (no ObjectId)
        video_data = get_video_by_string_id(video_id)
        
        if not video_data:
            return jsonify({"error": "Video not found"}), 404
        
        # Format the date
        processed_date = video_data.get('processed_at')
        if processed_date:
            if isinstance(processed_date, str):
                try:
                    date_obj = datetime.datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%b %d, %Y')
                except:
                    formatted_date = processed_date
            else:
                formatted_date = processed_date.strftime('%b %d, %Y')
        else:
            formatted_date = "Unknown date"
        
        # Prepare response
        response = {
            "id": video_id,  # Use the original string ID
            "filename": video_data.get('video_filename'),
            "title": os.path.splitext(video_data.get('video_filename', ''))[0].replace("_", " "),
            "date": formatted_date,
            "events": video_data.get('events', [])
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@api_blueprint.route('/event-frames/<filename>/<float:timestamp>', methods=['GET'])
def get_event_frames(filename, timestamp):
    """
    Get frames around an event timestamp for verification
    """
    video_path = os.path.join('video', filename)
    if not os.path.exists(video_path):
        return jsonify({"error": f"File {filename} not found in video folder"}), 404
    
    try:
        frames = extract_frames_around_timestamp(video_path, timestamp)
        
        return jsonify({
            "status": "success",
            "frames": frames
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500