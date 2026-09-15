import os
from flask import Flask, jsonify
from flask_cors import CORS
from routes import api_blueprint
from service import initialize_model

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register API blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Pre-initialize models on startup
    print("[nxt-sport-core] Initializing ML models...")
    try:
        initialize_model()
        print("[nxt-sport-core] Native ML model ready.")
    except Exception as e:
        print(f"[nxt-sport-core] Warning: Native model initialization error: {e}")

    try:
        from spivak_service import SpivakSoccerSpotter
        SpivakSoccerSpotter.get_instance()
        print("[nxt-sport-core] Spivak SoccerNet model ready.")
    except Exception as e:
        print(f"[nxt-sport-core] Warning: Spivak model initialization error: {e}")

    @app.route('/')
    def root():
        return jsonify({
            "service": "nxt-sport-core",
            "status": "online",
            "version": "2.0-stateless"
        })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=1111)