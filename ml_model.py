import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import cv2

class SoccerEventDetector:
    def __init__(self, model_path=None):
        """
        Initialize the Soccer Event Detector
        
        Args:
            model_path (str): Path to the saved model, if None, will create a new model
        """
    # Default classes - will be updated if a model with metadata is loaded
        self.classes = ['No Event', 'Goal', 'Attack']
        self.input_shape = (224, 224, 3)
        self.sequence_length = 10
        
        if model_path and os.path.exists(model_path):
            # First check if we have metadata
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    if "class_mapping" in metadata:
                        # Update classes from metadata
                        self.classes = [name for name in sorted(metadata["class_mapping"].keys())]
                        print(f"Loaded classes from metadata: {self.classes}")
                except Exception as e:
                    print(f"Error loading metadata: {e}")
            
            # Load the saved model
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self._build_model()
            print("New model created. Training required.")

    
    def _build_model(self):
        """
        Build a CNN+LSTM model for video classification
        """
        # Base model - using EfficientNetB0 (lightweight but powerful)
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create sequence model
        model = models.Sequential([
            # Time distributed wrapper for processing video frames
            layers.TimeDistributed(base_model, input_shape=(self.sequence_length, *self.input_shape)),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.TimeDistributed(layers.Dropout(0.3)),
            
            # Add recurrent layers to process the sequence
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(128),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for the model
        """
        # Resize to expected input size
        frame = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        
        # Convert to RGB if needed (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        frame = frame / 255.0
        
        return frame
    
    def predict_sequence(self, frames):
        """
        Predict the class of a sequence of frames
        
        Args:
            frames: List of frames to predict on
            
        Returns:
            Index of the predicted class
        """
        # Ensure we have the right number of frames
        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames)}")
        
        # Preprocess all frames
        preprocessed_frames = np.array([self.preprocess_frame(frame) for frame in frames])
        
        # Add batch dimension
        preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)
        
        # Make prediction
        prediction = self.model.predict(preprocessed_frames, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class_idx, confidence
    
    def save_model(self, path):
        """
        Save the model to disk
        """
        self.model.save(path)
        print(f"Model saved to {path}")

def download_pretrained_model():
    """
    Function to download a pretrained model if available
    Placeholder for actual download logic
    """
    model_dir = "models/soccer_events_model"
    
    # In a real application, you would implement logic to download 
    # a pre-trained model from a cloud storage or similar
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created directory {model_dir} for model storage")
        return None
    
    # Check if model exists
    if os.path.exists(f"{model_dir}/saved_model.pb"):
        return model_dir
    
    return None