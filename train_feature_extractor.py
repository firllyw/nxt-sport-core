"""
train_feature_extractor.py — Fine-tune ResNet-152 for soccer domain adaptation.

This is an OFFLINE tool — run it once on a machine with sufficient RAM/CPU,
then copy the resulting weights file to the production server.

Expected data directory structure:
    training_data/
        goal/          <- 30–50 MP4 clips, 5–10s each, showing a goal moment
        no_event/      <- 50–100 MP4 clips of regular play (passing, running, etc.)

YouTube sourcing tips:
    - Search "top 50 goals [league] [year]" — compilations work well
    - Use yt-dlp to download: yt-dlp -f "best[height<=480]" <url>
    - Trim individual goal moments to ~8s clips using FFmpeg:
        ffmpeg -ss 00:01:23 -i full_video.mp4 -t 8 -c copy goal/goal_001.mp4

Fine-tuning strategy (CPU-friendly):
    - Load ResNet-152 with ImageNet weights
    - Freeze ALL layers except conv5_block3 (last residual block, ~15 layers)
    - Add a temporary binary classification head (goal_moment vs no_event)
    - Train for up to 10 epochs with early stopping
    - Strip the classification head; save only backbone weights
    - Re-fit resnet_normalizer.pkl to the new feature distribution

Usage:
    python train_feature_extractor.py \\
        --training-dir ./training_data \\
        --output-weights ./models/feature_extractor/resnet_finetuned.h5 \\
        --epochs 10 --batch-size 8

    # With GPU (unfreezes 2 blocks instead of 1):
    python train_feature_extractor.py ... --gpu-mode
"""

import os
import sys
import cv2
import pickle
import argparse
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks as keras_callbacks
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FRAMES_PER_CLIP = 20       # Frames sampled per clip during dataset collection
TARGET_H = TARGET_W = 224  # ResNet-152 input size


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #

def _center_crop_224(frame):
    """Aspect-ratio-preserving center crop to 224×224 (matches SoccerNet prep)."""
    h, w = frame.shape[:2]
    if w > h:
        new_w = int(round(w * (224.0 / h)))
        resized = cv2.resize(frame, (new_w, 224))
        sx = max(0, (new_w - 224) // 2)
        return resized[:, sx:sx + 224]
    else:
        new_h = int(round(h * (224.0 / w)))
        resized = cv2.resize(frame, (224, new_h))
        sy = max(0, (new_h - 224) // 2)
        return resized[sy:sy + 224, :]


def load_frames_from_clip(video_path, n_frames=FRAMES_PER_CLIP):
    """Sample n_frames evenly spaced from a video clip. Returns list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}", flush=True)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = np.linspace(0, max(0, total - 1), min(n_frames, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        cropped = _center_crop_224(frame)
        frames.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def collect_dataset(training_dir, n_frames=FRAMES_PER_CLIP):
    """
    Load frames and binary labels from training_dir.

    Expected layout:
        training_dir/goal/      -> label 1
        training_dir/no_event/  -> label 0
    """
    class_config = {
        "goal":     1,
        "no_event": 0,
    }

    all_frames = []
    all_labels = []

    for class_name, label in class_config.items():
        class_dir = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[Dataset] Warning: '{class_dir}' not found — skipping class '{class_name}'", flush=True)
            continue

        video_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        ])
        print(f"[Dataset] '{class_name}' (label={label}): {len(video_files)} clips", flush=True)

        for vf in video_files:
            frames = load_frames_from_clip(os.path.join(class_dir, vf), n_frames=n_frames)
            if not frames:
                continue
            all_frames.extend(frames)
            all_labels.extend([label] * len(frames))
            print(f"  {vf}: {len(frames)} frames", flush=True)

    if not all_frames:
        raise ValueError(
            f"No training frames found in '{training_dir}'. "
            f"Expected subdirs: goal/, no_event/"
        )

    X = np.array(all_frames, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    n_goal = int(np.sum(y))
    n_bg = int(np.sum(1 - y))
    print(f"[Dataset] Total: {len(X)} frames — goal={n_goal}, no_event={n_bg}", flush=True)
    return X, y


# --------------------------------------------------------------------------- #
# Model building
# --------------------------------------------------------------------------- #

def build_fine_tune_model(gpu_mode=False):
    """
    Build ResNet-152 + binary classification head.

    CPU mode (default): unfreeze only conv5_block3 (~15 layers).
    GPU mode:           unfreeze conv5_block2 + conv5_block3 (~30 layers).

    Returns (full_model, backbone_model) where backbone produces 2048-dim features.
    """
    base = ResNet152(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # Freeze everything
    for layer in base.layers:
        layer.trainable = False

    # Unfreeze last residual block(s)
    target_blocks = ['conv5_block3'] if not gpu_mode else ['conv5_block2', 'conv5_block3']
    for layer in base.layers:
        if any(block in layer.name for block in target_blocks):
            layer.trainable = True

    n_trainable = sum(1 for l in base.layers if l.trainable)
    print(f"[Model] Trainable layers: {n_trainable} / {len(base.layers)} total", flush=True)

    # Classification head
    x = layers.Dense(256, activation='relu', name='ft_dense')(base.output)
    x = layers.Dropout(0.4, name='ft_dropout')(x)
    output = layers.Dense(1, activation='sigmoid', name='ft_classifier')(x)

    full_model = models.Model(inputs=base.input, outputs=output, name='resnet_finetuned')
    full_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return full_model, base


# --------------------------------------------------------------------------- #
# Post-training utilities
# --------------------------------------------------------------------------- #

def extract_backbone_weights(full_model, output_path):
    """
    Save only the ResNet-152 backbone weights (drops the classification head).
    The saved .h5 file is loaded by DomainAdaptedExtractor at inference time.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build a clean backbone-only model and transfer weights layer by layer
    backbone = ResNet152(weights=None, include_top=False, pooling='avg', input_shape=(224, 224, 3))
    trained_layers = {l.name: l for l in full_model.layers}

    transferred = 0
    for layer in backbone.layers:
        if layer.name in trained_layers and trained_layers[layer.name].get_weights():
            layer.set_weights(trained_layers[layer.name].get_weights())
            transferred += 1

    backbone.save_weights(output_path)
    print(f"[Export] Backbone weights saved to: {output_path} ({transferred} layers transferred)", flush=True)


def refit_normalizer(full_model, X_raw, normalizer_path):
    """
    Re-fit MaxAbsScaler on a sample of features from the fine-tuned backbone.
    Updates resnet_normalizer.pkl so Spivak's normalisation step stays calibrated.
    """
    print("[Normalizer] Refitting on new feature distribution...", flush=True)

    # Feature extractor = full model minus the classification head
    feature_model = models.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer('avg_pool').output,
        name='feature_extractor_tmp'
    )

    # Sample up to 500 frames for fitting
    n_sample = min(500, len(X_raw))
    idx = np.random.choice(len(X_raw), n_sample, replace=False)
    X_sub = preprocess_input(X_raw[idx].copy())
    features = feature_model.predict(X_sub, batch_size=16, verbose=0)

    scaler = MaxAbsScaler()
    scaler.fit(features)

    os.makedirs(os.path.dirname(normalizer_path), exist_ok=True)
    with open(normalizer_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"[Normalizer] Saved to: {normalizer_path}", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-152 feature extractor for soccer domain adaptation"
    )
    parser.add_argument(
        "--training-dir", default="./training_data",
        help="Root dir containing goal/ and no_event/ subdirectories of MP4 clips"
    )
    parser.add_argument(
        "--output-weights", default="./models/feature_extractor/resnet_finetuned.h5",
        help="Output path for fine-tuned backbone weights (.h5)"
    )
    parser.add_argument(
        "--normalizer-output", default="./models/spivak_soccernet/resnet_normalizer.pkl",
        help="Path to update the Spivak normalizer (re-fitted to new features)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (keep small for CPU)")
    parser.add_argument("--frames-per-clip", type=int, default=FRAMES_PER_CLIP,
                        help="Frames sampled from each training clip")
    parser.add_argument("--gpu-mode", action="store_true",
                        help="Unfreeze 2 residual blocks instead of 1 (use when GPU is available)")
    args = parser.parse_args()

    print("=" * 60)
    print("nxt-sport-core — Feature Extractor Fine-Tuning")
    print(f"  Training dir   : {args.training_dir}")
    print(f"  Output weights : {args.output_weights}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Frames/clip    : {args.frames_per_clip}")
    print(f"  GPU mode       : {args.gpu_mode}")
    print("=" * 60)

    # 1. Load dataset
    X, y = collect_dataset(args.training_dir, n_frames=args.frames_per_clip)

    # 2. Split and preprocess
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    X_train_pp = preprocess_input(X_train.copy())
    X_val_pp = preprocess_input(X_val.copy())
    print(f"[Train] Split: {len(X_train)} train / {len(X_val)} val frames", flush=True)

    # 3. Build model
    full_model, _base = build_fine_tune_model(gpu_mode=args.gpu_mode)

    # 4. Train
    cb = [
        keras_callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True, verbose=1
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2,
            min_lr=1e-7, verbose=1
        ),
    ]

    history = full_model.fit(
        X_train_pp, y_train,
        validation_data=(X_val_pp, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1
    )

    final_val_acc = float(history.history['val_accuracy'][-1])
    epochs_run = len(history.history['loss'])

    # 5. Extract and save backbone weights (drop classification head)
    extract_backbone_weights(full_model, args.output_weights)

    # 6. Refit normalizer
    refit_normalizer(full_model, X, args.normalizer_output)

    # 7. Save training metadata
    meta_path = os.path.join(os.path.dirname(args.output_weights), "training_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "training_dir": args.training_dir,
            "epochs_run": epochs_run,
            "epochs_requested": args.epochs,
            "final_val_accuracy": final_val_acc,
            "gpu_mode": args.gpu_mode,
            "frames_per_clip": args.frames_per_clip,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }, f, indent=2)

    print()
    print("=" * 60)
    print(f"Fine-tuning complete!")
    print(f"  Val accuracy  : {final_val_acc:.3f}")
    print(f"  Weights saved : {args.output_weights}")
    print(f"  Normalizer    : {args.normalizer_output}")
    print()
    print("Next steps:")
    print("  1. Copy resnet_finetuned.h5 to the production server")
    print("  2. Restart nxt-sport-core (weights auto-loaded at startup)")
    print("  3. Test with a known match video and compare event counts")
    print("  4. For further gains: collect clips from YOUR own footage and re-run")
    print("=" * 60)


if __name__ == "__main__":
    main()
