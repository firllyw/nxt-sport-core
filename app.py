from flask import Flask, send_from_directory
from routes import api_blueprint
import os
from ml_model import SoccerEventDetector
from flask_cors import CORS

# Global model variable
EVENT_DETECTOR = None

def initialize_model():
    global EVENT_DETECTOR
    model_path = "models/soccer_events_model"
    if os.path.exists(model_path):
        EVENT_DETECTOR = SoccerEventDetector(model_path)
    else:
        print("WARNING: No trained model found")
        EVENT_DETECTOR = SoccerEventDetector()
    return EVENT_DETECTOR

def create_app():
    app = Flask(__name__, static_url_path='/static')
    CORS(app)  # Enable CORS for all routes
    # Register the API blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Create necessary directories
    os.makedirs('./static/frames', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./training/goals', exist_ok=True)
    os.makedirs('./training/attacks', exist_ok=True)
    os.makedirs('./training/no_events', exist_ok=True)
    os.makedirs('./video', exist_ok=True)
    
    # Initialize model
    initialize_model()
    
    # Serve static files
    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory('static', path)
    
    @app.route('/video-stream/<path:filename>')
    def serve_video_stream(filename):
        """
        Stream a video file
        """
        video_dir = os.path.join(app.root_path, 'video')
        
        # Check if file exists
        if not os.path.exists(os.path.join(video_dir, filename)):
            return "Video not found", 404
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Set content type based on file extension
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska'
        }
        
        return send_from_directory(
            video_dir, 
            filename,
            mimetype=mime_types.get(ext.lower(), 'application/octet-stream')
        )
    
    @app.route('/upload')
    def upload_form():
        """Serve the upload form HTML"""
        return send_from_directory('static', 'upload_form.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=1111)