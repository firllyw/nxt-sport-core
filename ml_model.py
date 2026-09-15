import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import cv2

class SoccerEventDetector:
    def __init__(self, model_path="models/soccer_events_model", for_training=False, num_classes=3):
        """
        Initialize the Soccer Event Detector.
        If for_training is True, creates a trainable Keras model.
        Otherwise loads TensorFlow SavedModel for fast production inference.
        """
        self.class_map = {
            0: "Attack",
            1: "Goal",
            2: "No Event"
        }
        self.classes = ['attacks', 'goals', 'no_events']
        self.input_shape = (224, 224, 3)
        self.sequence_length = 10
        self.infer = None
        self.model = None
        self.model_path = model_path

        if for_training:
            self.model = self._build_model(num_classes=num_classes)
            print(f"[SoccerEventDetector] Trainable Keras model built for {num_classes} classes.")
            return

        if model_path and os.path.exists(model_path):
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                    if "class_mapping" in meta:
                        raw_map = {idx: name for name, idx in meta["class_mapping"].items()}
                        name_pretty = {"attacks": "Attack", "goals": "Goal", "no_events": "No Event"}
                        self.class_map = {idx: name_pretty.get(name, name.capitalize()) for idx, name in raw_map.items()}
                        self.classes = [name for name in sorted(meta["class_mapping"].keys())]
                        print(f"[SoccerEventDetector] Loaded class mapping: {self.class_map}")
                except Exception as e:
                    print(f"[SoccerEventDetector] Error loading metadata: {e}")

            print(f"[SoccerEventDetector] Loading TensorFlow SavedModel from {model_path}...")
            imported = tf.saved_model.load(model_path)
            self.infer = imported.signatures["serving_default"]
            print("[SoccerEventDetector] Model loaded successfully for inference.")
        else:
            self.model = self._build_model(num_classes=num_classes)
            print("[SoccerEventDetector] Model path not found. Initialized new trainable model.")

    def _build_model(self, num_classes=3):
        """
        Build EfficientNetB0 + LSTM video classification model
        """
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = False

        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, *self.input_shape)),
            layers.TimeDistributed(base_model),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.TimeDistributed(layers.Dropout(0.3)),
            layers.LSTM(256, return_sequences=True),
            layers.LSTM(128),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def save_model(self, path):
        """
        Save the model in TensorFlow SavedModel format
        """
        if self.model is None:
            raise RuntimeError("No trainable Keras model available to save.")
        os.makedirs(path, exist_ok=True)
        self.model.export(path)
        print(f"[SoccerEventDetector] Model exported to {path}")

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame: resize, convert to RGB, normalize 0-1
        """
        frame_resized = cv2.resize(frame, (self.input_shape[0], self.input_shape[1]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype(np.float32) / 255.0

    def predict_sequence(self, frames):
        """
        Predict event class for a sequence of 10 frames
        """
        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames)}")

        processed = np.array([self.preprocess_frame(f) for f in frames], dtype=np.float32)

        if self.infer is not None:
            batch_input = tf.constant(np.expand_dims(processed, axis=0))
            outputs = self.infer(batch_input)
            key = list(outputs.keys())[0]
            probs = outputs[key].numpy()[0]
        elif self.model is not None:
            batch_input = np.expand_dims(processed, axis=0)
            probs = self.model.predict(batch_input, verbose=0)[0]
        else:
            raise RuntimeError("Neither SavedModel infer nor Keras model is loaded.")

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        class_name = self.class_map.get(predicted_idx, "Unknown")

        return predicted_idx, confidence, class_name