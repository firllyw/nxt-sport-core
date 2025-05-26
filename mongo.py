from pymongo import MongoClient
import datetime
import os

# MongoDB connection details
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://root:secret@localhost:27017/')
DB_NAME = os.environ.get('MONGO_DB', 'soccer_events')
COLLECTION_NAME = 'events'

def get_mongodb_client():
    """
    Get MongoDB client instance
    """
    client = MongoClient(MONGO_URI)
    return client

def save_events_to_db(video_filename, events):
    """
    Save extracted events to MongoDB
    
    Args:
        video_filename (str): Name of the video file
        events (list): List of event dictionaries with timestamp and event type
    
    Returns:
        str: ID of the inserted document
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Create document to insert
    document = {
        "video_filename": video_filename,
        "processed_at": datetime.datetime.utcnow(),
        "events": events
    }
    
    # Insert document
    result = collection.insert_one(document)
    
    # Close connection
    client.close()
    
    return str(result.inserted_id)

def get_all_videos_with_events():
    """
    Get a list of all videos with their event counts
    
    Returns:
        list: List of dictionaries with video details and event counts
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    pipeline = [
        {
            "$project": {
                "video_filename": 1,
                "processed_at": 1,
                "event_name": 1,
                "event_place": 1,
                "event_date": 1,
                "goal_count": {
                    "$size": {
                        "$filter": {
                            "input": "$events",
                            "as": "event",
                            "cond": {"$eq": ["$$event.event_type", "Goal"]}
                        }
                    }
                },
                "attack_count": {
                    "$size": {
                        "$filter": {
                            "input": "$events",
                            "as": "event",
                            "cond": {"$eq": ["$$event.event_type", "Attack"]}
                        }
                    }
                }
            }
        }
    ]
    
    result = list(collection.aggregate(pipeline))
    
    # Close connection
    client.close()
    
    return result

def get_video_by_string_id(id_string):
    """
    Get a video by string ID without using ObjectId
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Fetch all documents (this is inefficient but will work for small collections)
    all_docs = list(collection.find())
    
    # Manually search for the document with matching string ID
    result = None
    for doc in all_docs:
        if str(doc.get('_id')) == id_string:
            result = doc
            break
    
    client.close()
    return result

def save_video_metadata(metadata):
    """
    Save initial video metadata to MongoDB
    
    Args:
        metadata (dict): Video metadata including ID, filename, event details
    
    Returns:
        str: ID of the inserted document
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Insert document
    result = collection.insert_one(metadata)
    
    # Close connection
    client.close()
    
    return str(result.inserted_id)

def update_video_status(video_id, status, error=None):
    """
    Update video processing status
    
    Args:
        video_id (str): Video ID
        status (str): Processing status ('processing', 'completed', 'failed')
        error (str, optional): Error message if status is 'failed'
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Update document
    update_data = {"processing_status": status}
    if error:
        update_data["processing_error"] = error
    
    collection.update_one({"video_id": video_id}, {"$set": update_data})
    
    # Close connection
    client.close()

def update_video_events(video_id, events, duration):
    """
    Update video document with extracted events and duration
    
    Args:
        video_id (str): Video ID
        events (list): List of event dictionaries
        duration (float): Video duration in seconds
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Format duration as string (MM:SS)
    mins = int(duration // 60)
    secs = int(duration % 60)
    duration_str = f"{mins}:{secs:02d}"
    
    # Update document
    collection.update_one(
        {"video_id": video_id}, 
        {"$set": {
            "events": events,
            "duration": duration,
            "duration_str": duration_str,
            "processed_at": datetime.datetime.utcnow()
        }}
    )
    
    # Close connection
    client.close()

def get_video_processing_status(video_id):
    """
    Get the processing status of a video
    
    Args:
        video_id (str): Video ID
    
    Returns:
        dict: Status information or None if not found
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Find document
    document = collection.find_one(
        {"video_id": video_id},
        {"_id": 0, "video_id": 1, "processing_status": 1, "processing_error": 1}
    )
    
    # Close connection
    client.close()
    
    return document

def get_video_by_id(video_id):
    """
    Get a video by its ID
    
    Args:
        video_id: MongoDB ObjectId or string ID
    
    Returns:
        dict: Video document or None if not found
    """
    client = get_mongodb_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Try to find by _id if ObjectId
    if isinstance(video_id, ObjectId):
        document = collection.find_one({"_id": video_id})
    else:
        # Try finding by video_id or _id as string
        document = collection.find_one({"$or": [
            {"video_id": video_id},
            {"_id": video_id}
        ]})
    
    # Close connection
    client.close()
    
    return document