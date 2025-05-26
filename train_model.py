import os
import cv2
import numpy as np
import tensorflow as tf
from ml_model import SoccerEventDetector
import random
from collections import deque
import argparse
import json
from datetime import datetime

def extract_frame_sequences(video_path, sequence_length=10, num_sequences=5):
    """
    Extract multiple sequences of frames from a video
    
    Args:
        video_path: Path to the video file
        sequence_length: Number of frames in each sequence
        num_sequences: Number of sequences to extract
        
    Returns:
        List of frame sequences (each is a numpy array)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video is too short, return empty list
    if frame_count < sequence_length:
        print(f"Video {video_path} is too short ({frame_count} frames)")
        cap.release()
        return []
    
    # Calculate possible start positions
    max_start = frame_count - sequence_length
    
    # Randomly select start positions
    if max_start <= num_sequences:
        # If video is short, use all possible sequences with overlap
        start_positions = list(range(0, max_start+1))
        # If we need more sequences than available, duplicate some
        while len(start_positions) < num_sequences:
            start_positions.append(random.randint(0, max_start))
    else:
        # Randomly sample start positions without replacement
        start_positions = sorted(random.sample(range(0, max_start+1), num_sequences))
    
    sequences = []
    
    for start_pos in start_positions:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        
        # Read sequence_length frames
        frame_sequence = []
        for _ in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Normalize
            
            frame_sequence.append(frame)
        
        # Only add complete sequences
        if len(frame_sequence) == sequence_length:
            sequences.append(np.array(frame_sequence))
    
    cap.release()
    return sequences

def prepare_training_data(training_dir, sequence_length=10, sequences_per_video=5):
    """
    Prepare training data from labeled video directories
    
    Args:
        training_dir: Root directory containing class subdirectories
        sequence_length: Number of frames per sequence
        sequences_per_video: Number of sequences to extract from each video
        
    Returns:
        X_train: Training sequences
        y_train: Labels
        class_mapping: Dictionary mapping class indices to names
    """
    # Class mapping
    class_dirs = [d for d in os.listdir(training_dir) 
                  if os.path.isdir(os.path.join(training_dir, d))]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {training_dir}")
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Map class names to indices
    class_mapping = {name: idx for idx, name in enumerate(sorted(class_dirs))}
    
    X_train = []
    y_train = []
    
    # Process each class directory
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(training_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) 
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Found {len(video_files)} videos for class '{class_name}'")
        
        if not video_files:
            print(f"Warning: No videos found in {class_dir}")
            continue
        
        # Process each video in the class directory
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            print(f"Processing {video_path}...")
            
            # Extract sequences from the video
            sequences = extract_frame_sequences(
                video_path, 
                sequence_length=sequence_length,
                num_sequences=sequences_per_video
            )
            
            # Add sequences and labels
            X_train.extend(sequences)
            y_train.extend([class_idx] * len(sequences))
            
            print(f"  Extracted {len(sequences)} sequences")
    
    # Convert to numpy arrays
    if not X_train:
        raise ValueError("No valid sequences were extracted from the videos")
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Convert labels to one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(class_mapping))
    
    print(f"Final dataset: {X_train.shape[0]} sequences, {y_train_one_hot.shape[1]} classes")
    
    # Invert class mapping for reporting
    idx_to_class = {idx: name for name, idx in class_mapping.items()}
    
    # Report class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  Class '{idx_to_class[idx]}': {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train_one_hot, class_mapping

def train_model(training_dir, model_dir, epochs=10, batch_size=8):
    """
    Train the soccer event detection model
    
    Args:
        training_dir: Directory containing class subdirectories with videos
        model_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Record training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "training_dir": training_dir,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    # Prepare training data
    try:
        X_train, y_train, class_mapping = prepare_training_data(training_dir)
        
        # Add dataset info to metadata
        metadata["num_samples"] = len(X_train)
        metadata["class_mapping"] = class_mapping
        metadata["input_shape"] = X_train.shape[1:]
        
        # Save metadata
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Check if we already have a trained model to continue training
        if os.path.exists(os.path.join(model_dir, "saved_model.pb")):
            print(f"Loading existing model from {model_dir}")
            detector = SoccerEventDetector(model_dir)
            print("Continuing training on existing model")
        else:
            print("Creating new model")
            detector = SoccerEventDetector()
        
        # Update model classes if needed
        detector.classes = [class_name for class_name in sorted(class_mapping.keys())]
        
        # Train the model
        print(f"Training model with {len(X_train)} sequences...")
        history = detector.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True
        )
        
        # Save training history
        with open(os.path.join(model_dir, "training_history.json"), "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            history_dict = {}
            for k, v in history.history.items():
                history_dict[k] = [float(val) for val in v]
            json.dump(history_dict, f, indent=2)
        
        # Save the trained model
        detector.save_model(model_dir)
        print(f"Model trained and saved to {model_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train soccer event detection model")
    parser.add_argument("--training-dir", type=str, default="./training",
                        help="Directory containing class subdirectories with training videos")
    parser.add_argument("--model-dir", type=str, default="./models/soccer_events_model",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    print("Starting model training...")
    train_model(args.training_dir, args.model_dir, args.epochs, args.batch_size)
    print("Training complete!")

if __name__ == "__main__":
    main()