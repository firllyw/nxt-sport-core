"""
feature_extractor.py — Domain-adapted ResNet-152 feature extractor.

This module wraps ResNet-152 with support for loading fine-tuned weights
produced by train_feature_extractor.py.

When fine-tuned weights are present at models/feature_extractor/resnet_finetuned.h5,
they are applied on top of ImageNet weights to produce 2048-dim feature vectors
that are closer to the SoccerNet distribution expected by the Spivak model.

Without fine-tuned weights the extractor silently falls back to generic ImageNet
ResNet-152 (identical to the previous behaviour — no regression).

The two-hop fine-tuning strategy:
  Hop 1: Fine-tune on YouTube broadcast soccer clips → closes gap with SoccerNet
  Hop 2: (future) Fine-tune on YOUR own match footage → closes remaining domain gap

Usage from spivak_service.py:
    from feature_extractor import DomainAdaptedExtractor
    extractor = DomainAdaptedExtractor()
    features, duration = extractor.extract_features(video_path, fps=2.0)
"""

import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input


DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "models", "feature_extractor", "resnet_finetuned.h5"
)


class DomainAdaptedExtractor:
    """
    ResNet-152 feature extractor with optional domain-adapted weights.

    Attributes:
        resnet:        The underlying Keras ResNet-152 model (available for
                       direct access by callers that need it, e.g. spivak_service).
        is_finetuned:  True if fine-tuned weights were successfully loaded.
        feature_dim:   Output feature dimensionality (2048).
    """

    def __init__(self, weights_path=None, feature_dim=2048):
        """
        Args:
            weights_path: Path to .h5 fine-tuned backbone weights.
                          Defaults to models/feature_extractor/resnet_finetuned.h5.
            feature_dim:  Expected output dimension (must remain 2048 for Spivak compat).
        """
        self.feature_dim = feature_dim

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH

        print("[Extractor] Loading ResNet-152 (ImageNet base weights)...", flush=True)
        self.resnet = ResNet152(weights='imagenet', include_top=False, pooling='avg')
        self.is_finetuned = False

        if os.path.exists(weights_path):
            try:
                self.resnet.load_weights(weights_path)
                print(f"[Extractor] Fine-tuned weights loaded from: {weights_path}", flush=True)
                self.is_finetuned = True
            except Exception as exc:
                print(
                    f"[Extractor] Warning: could not load fine-tuned weights ({exc}). "
                    f"Falling back to ImageNet.",
                    flush=True
                )
        else:
            print(
                f"[Extractor] No fine-tuned weights found at {weights_path}. "
                f"Using generic ImageNet weights. "
                f"Run train_feature_extractor.py to improve accuracy.",
                flush=True
            )

    def extract_features(self, video_path, fps=2.0, batch_size=32, progress_callback=None):
        """
        Sample frames from video at `fps` and extract 2048-dim ResNet-152 features.

        Uses aspect-ratio-preserving center crop to 224×224, matching the
        preprocessing used during SoccerNet feature extraction.

        Args:
            video_path:        Path to input MP4.
            fps:               Target sampling frame rate (2.0 matches Spivak's input spec).
            batch_size:        Frames per inference batch (lower for CPU memory limits).
            progress_callback: Optional callable(pct: int, msg: str) for progress updates.

        Returns:
            Tuple of (features, duration):
                features: numpy array of shape (N, 2048), dtype float32
                duration: float, video duration in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[Extractor] Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / orig_fps if orig_fps > 0 else 0.0

        frame_interval = orig_fps / fps if orig_fps > 0 else 15.0
        total_samples = max(1, int(round(duration * fps)))

        features_list = []
        frame_batch = []
        frame_idx = 0
        next_target_frame = 0.0
        sample_count = 0
        last_reported_pct = 0

        print(
            f"[Extractor] duration={duration:.1f}s, orig_fps={orig_fps:.1f}, "
            f"target_fps={fps}, expected_samples={total_samples}, "
            f"finetuned={self.is_finetuned}",
            flush=True
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= int(round(next_target_frame)):
                # Aspect-ratio-preserving center crop to 224×224
                h, w = frame.shape[:2]
                if w > h:
                    new_w = int(round(w * (224.0 / h)))
                    resized = cv2.resize(frame, (new_w, 224))
                    sx = max(0, (new_w - 224) // 2)
                    frame_cropped = resized[:, sx:sx + 224]
                else:
                    new_h = int(round(h * (224.0 / w)))
                    resized = cv2.resize(frame, (224, new_h))
                    sy = max(0, (new_h - 224) // 2)
                    frame_cropped = resized[sy:sy + 224, :]

                frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame_rgb)
                sample_count += 1
                next_target_frame += frame_interval

                if len(frame_batch) >= batch_size:
                    batch_arr = np.array(frame_batch, dtype=np.float32)
                    preprocessed = preprocess_input(batch_arr)
                    batch_feats = self.resnet.predict(preprocessed, verbose=0)
                    features_list.append(batch_feats)
                    frame_batch = []

                    if progress_callback and total_samples > 0:
                        pct = min(80, int((sample_count / total_samples) * 80))
                        if pct > last_reported_pct:
                            last_reported_pct = pct
                            progress_callback(
                                pct,
                                f"Mengekstrak fitur ResNet-152... ({pct}%)"
                            )

            frame_idx += 1

        # Flush remaining frames in batch
        if frame_batch:
            batch_arr = np.array(frame_batch, dtype=np.float32)
            preprocessed = preprocess_input(batch_arr)
            batch_feats = self.resnet.predict(preprocessed, verbose=0)
            features_list.append(batch_feats)

        cap.release()

        if not features_list:
            return np.zeros((0, self.feature_dim), dtype=np.float32), duration

        all_features = np.vstack(features_list)
        print(
            f"[Extractor] Extracted {all_features.shape[0]} vectors, "
            f"mean={np.mean(all_features):.3f}, std={np.std(all_features):.3f}",
            flush=True
        )
        return all_features, duration
