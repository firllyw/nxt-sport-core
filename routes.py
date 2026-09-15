import os
import threading
import requests
from flask import Blueprint, request, jsonify, current_app
from service import process_match_video_pipeline, initialize_model
from minio_storage import ensure_bucket

api_blueprint = Blueprint('api', __name__)

# In-memory jobs tracking
JOBS = {}
JOBS_LOCK = threading.Lock()

def send_webhook(callback_url, payload):
    """
    Send webhook update to OpenList backend
    """
    if not callback_url:
        return
    try:
        requests.post(callback_url, json=payload, timeout=10)
    except Exception as e:
        print(f"[Webhook] Failed to notify {callback_url}: {e}")

def run_job_in_background(job_id, gdrive_url, event_id, match_id, callback_url, options):
    """
    Worker executing the end-to-end pipeline
    """
    print(f"[{job_id}] Starting video job for event={event_id}, match={match_id}")
    
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "QUEUED",
            "progress": 0,
            "step": "Job diantrekan...",
            "highlights": [],
            "error": None
        }

    def progress_callback(status, pct, msg):
        with JOBS_LOCK:
            JOBS[job_id]["status"] = status
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["step"] = msg

        # Send webhook progress
        send_webhook(callback_url, {
            "job_id": job_id,
            "event_id": event_id,
            "match_id": match_id,
            "status": status,
            "progress": pct,
            "current_step": msg
        })

    try:
        result = process_match_video_pipeline(
            job_id=job_id,
            gdrive_url_or_path=gdrive_url,
            event_id=event_id,
            match_id=match_id,
            options=options,
            progress_callback=progress_callback
        )
        
        num_hl = len(result['highlights'])
        step_msg = f"Selesai ({num_hl} momen ditemukan)" if num_hl > 0 else "Selesai (0 momen terdeteksi dengan sensitivitas ini)"
        
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "COMPLETED"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["step"] = step_msg
            JOBS[job_id]["duration"] = result["duration"]
            JOBS[job_id]["highlights"] = result["highlights"]

        # Send webhook completion
        send_webhook(callback_url, {
            "job_id": job_id,
            "event_id": event_id,
            "match_id": match_id,
            "status": "COMPLETED",
            "progress": 100,
            "current_step": step_msg,
            "video_duration": result["duration"],
            "highlights": result["highlights"]
        })
        print(f"[{job_id}] Completed successfully with {num_hl} highlights!", flush=True)

    except Exception as e:
        err_msg = str(e)
        print(f"[{job_id}] Job error: {err_msg}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "FAILED"
            JOBS[job_id]["error"] = err_msg
            JOBS[job_id]["step"] = f"Error: {err_msg}"

        # Send webhook failure
        send_webhook(callback_url, {
            "job_id": job_id,
            "event_id": event_id,
            "match_id": match_id,
            "status": "FAILED",
            "error": err_msg
        })

@api_blueprint.route('/health', methods=['GET'])
def health():
    minio_ok = False
    try:
        minio_ok = ensure_bucket()
    except Exception:
        minio_ok = False
        
    return jsonify({
        "status": "ok",
        "service": "nxt-sport-core",
        "minio_ready": minio_ok
    }), 200

@api_blueprint.route('/jobs/process', methods=['POST'])
def create_processing_job():
    """
    Enqueue a video processing job
    Payload:
    {
        "job_id": "...",
        "gdrive_url": "...",
        "event_id": "...",
        "match_id": "...",
        "callback_url": "...",
        "options": {}
    }
    """
    data = request.get_json() or {}
    job_id = data.get("job_id")
    gdrive_url = data.get("gdrive_url")
    event_id = data.get("event_id")
    match_id = data.get("match_id", "general")
    callback_url = data.get("callback_url")
    options = data.get("options", {})

    if not job_id or not gdrive_url:
        return jsonify({"error": "job_id and gdrive_url are required"}), 400

    thread = threading.Thread(
        target=run_job_in_background,
        args=(job_id, gdrive_url, event_id, match_id, callback_url, options)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "queued",
        "job_id": job_id,
        "message": "Processing job queued successfully"
    }), 202

@api_blueprint.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404
        
    return jsonify(job), 200