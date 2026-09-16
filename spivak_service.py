import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input

# 17 SoccerNet classes
SOCCERNET_CLASSES = {
    0: "Penalty",
    1: "Kick-off",
    2: "Goal",
    3: "Substitution",
    4: "Offside",
    5: "Shots on target",
    6: "Shots off target",
    7: "Clearance",
    8: "Ball out of play",
    9: "Throw-in",
    10: "Foul",
    11: "Indirect free-kick",
    12: "Direct free-kick",
    13: "Corner",
    14: "Yellow card",
    15: "Red card",
    16: "Yellow->red card"
}

# Target mappings to OpenList highlight event types
EVENT_MAPPING = {
    2: "GOAL",              # Goal
    0: "PENALTY",           # Penalty
    5: "ATTACK",            # Shots on target
    6: "ATTACK",            # Shots off target
    14: "CARD",             # Yellow card
    15: "CARD"              # Red card
}

class SpivakSoccerSpotter:
    _instance = None

    @classmethod
    def get_instance(cls, model_dir=None):
        if cls._instance is None:
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(__file__), "models", "spivak_soccernet")
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.fps = 2.0  # Spivak operates at 2.0 FPS (1 frame per 0.5 second)
        self.chunk_size = 224  # 224 steps = 112 seconds
        self.feature_dim = 2048

        print(f"[Spivak] Initializing feature extractor (domain-adapted ResNet-152)...")
        try:
            from feature_extractor import DomainAdaptedExtractor
            self._extractor = DomainAdaptedExtractor()
            self.resnet = self._extractor.resnet  # keep self.resnet for normalizer compat
        except ImportError:
            print("[Spivak] feature_extractor module not found — using raw ResNet-152.", flush=True)
            self._extractor = None
            self.resnet = ResNet152(weights='imagenet', include_top=False, pooling='avg')

        print(f"[Spivak] Loading normalizer from {model_dir}...")
        normalizer_path = os.path.join(model_dir, "resnet_normalizer.pkl")
        if os.path.exists(normalizer_path):
            with open(normalizer_path, "rb") as f:
                self.scaler = pickle.load(f)
            self.max_abs = getattr(self.scaler, "max_abs_", None)
            if self.max_abs is None:
                self.max_abs = getattr(self.scaler, "scale_", None)
        else:
            print("[Spivak] Warning: normalizer not found, using unit scaling.")
            self.max_abs = np.ones((self.feature_dim,), dtype=np.float32)

        print(f"[Spivak] Loading Spivak SavedModel from {model_dir}...")
        spivak_model = tf.saved_model.load(model_dir)
        self.infer = spivak_model.signatures['serving_default']
        print("[Spivak] Spivak Action Spotting model loaded successfully.")

    def extract_features(self, video_path, batch_size=32, progress_callback=None):
        """
        Sample frames from video at 2 FPS and extract ResNet-152 2048-dim features
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / orig_fps if orig_fps > 0 else 0.0

        # Sample timestamps at 2.0 FPS using sequential video reading (avoids cap.set seek drift)
        frame_interval = orig_fps / self.fps if orig_fps > 0 else 15.0
        total_samples = max(1, int(round(duration * self.fps)))
        
        features_list = []
        frame_batch = []
        
        frame_idx = 0
        next_target_frame = 0.0
        sample_count = 0
        last_reported_pct = 0
        
        print(f"[Spivak] Extracting frames: total_duration={duration:.1f}s, orig_fps={orig_fps:.1f}, expected_samples={total_samples}", flush=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx >= int(round(next_target_frame)):
                # Aspect-ratio preserving center crop to 224x224 (matching SoccerNet DevKit)
                h, w = frame.shape[:2]
                if w > h:
                    new_w = int(round(w * (224.0 / h)))
                    resized = cv2.resize(frame, (new_w, 224))
                    start_x = max(0, (new_w - 224) // 2)
                    frame_cropped = resized[:, start_x:start_x + 224]
                else:
                    new_h = int(round(h * (224.0 / w)))
                    resized = cv2.resize(frame, (224, new_h))
                    start_y = max(0, (new_h - 224) // 2)
                    frame_cropped = resized[start_y:start_y + 224, :]

                # Convert BGR -> RGB for Keras preprocess_input
                frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame_rgb)
                sample_count += 1
                next_target_frame += frame_interval
                
                if len(frame_batch) >= batch_size:
                    batch_arr = np.array(frame_batch, dtype=np.float32)
                    preprocessed = preprocess_input(batch_arr)
                    batch_features = self.resnet.predict(preprocessed, verbose=0)
                    features_list.append(batch_features)
                    frame_batch = []
                    
                    if progress_callback and total_samples > 0:
                        pct = min(80, int((sample_count / total_samples) * 80))
                        if pct > last_reported_pct:
                            last_reported_pct = pct
                            progress_callback(pct, f"Mengekstrak visual ResNet-152... ({pct}%)")

            frame_idx += 1

        # Process any remaining frames
        if frame_batch:
            batch_arr = np.array(frame_batch, dtype=np.float32)
            preprocessed = preprocess_input(batch_arr)
            batch_features = self.resnet.predict(preprocessed, verbose=0)
            features_list.append(batch_features)
            frame_batch = []

        cap.release()

        if not features_list:
            return np.zeros((0, self.feature_dim), dtype=np.float32), duration

        all_features = np.vstack(features_list)
        print(f"[Spivak] Visual features extracted: shape={all_features.shape}, mean_val={np.mean(all_features):.2f}", flush=True)
        return all_features, duration

    def spot_actions(self, features, min_confidence=0.18, stride=56, progress_callback=None):
        """
        Run Spivak 1D Temporal U-Net across sliding chunks of 224 steps.
        Uses max-pooling across chunks to preserve sharp prediction peaks
        and intelligent multi-signal fusion (Shots on target + Goal peaks)
        to identify real matchday goals, attacks, and penalties while rejecting noise.
        """
        num_timesteps = len(features)
        if num_timesteps == 0:
            return []

        # Normalize features with trained scaler
        if self.max_abs is not None:
            norm_features = features / (self.max_abs + 1e-7)
        else:
            norm_features = features

        # Use maximum-pooling across sliding chunks to avoid zero-pad edge dilution
        final_probs = np.zeros((num_timesteps, 17), dtype=np.float32)

        # Sliding window over timesteps
        chunk_starts = list(range(0, max(1, num_timesteps - self.chunk_size + 1), stride))
        if not chunk_starts or chunk_starts[-1] + self.chunk_size < num_timesteps:
            chunk_starts.append(max(0, num_timesteps - self.chunk_size))

        for c_idx, start_idx in enumerate(chunk_starts):
            end_idx = min(num_timesteps, start_idx + self.chunk_size)
            chunk = norm_features[start_idx:end_idx]
            actual_len = len(chunk)

            # Pad if shorter than chunk_size
            if actual_len < self.chunk_size:
                pad_width = self.chunk_size - actual_len
                chunk_padded = np.pad(chunk, ((0, pad_width), (0, 0)), mode='constant')
            else:
                chunk_padded = chunk

            # Input shape expected by Spivak: (1, 224, 2048, 1)
            inp = np.expand_dims(chunk_padded, axis=(0, 3)).astype(np.float32)
            outputs = self.infer(tf.constant(inp))
            logits = outputs['confidence_logits'].numpy()
            probs = tf.nn.sigmoid(logits).numpy()[0, :, :, 0]  # shape: (224, 17)

            # Update final_probs using element-wise maximum over overlapping chunks
            final_probs[start_idx:end_idx] = np.maximum(
                final_probs[start_idx:end_idx],
                probs[:actual_len]
            )

            if progress_callback:
                pct = 80 + int(((c_idx + 1) / len(chunk_starts)) * 18)  # 80-98%
                progress_callback(pct, f"Inference Spivak Temporal U-Net... ({pct}%)")

        # Key class probabilities
        goal_probs = final_probs[:, 2]         # Class 2: Goal
        shot_on_probs = final_probs[:, 5]      # Class 5: Shots on target
        shot_off_probs = final_probs[:, 6]     # Class 6: Shots off target
        penalty_probs = final_probs[:, 0]      # Class 0: Penalty
        yellow_probs = final_probs[:, 14]      # Class 14: Yellow card
        red_probs = final_probs[:, 15]         # Class 15: Red card

        # Log peak diagnostics
        print(f"[Spivak] Inferred {num_timesteps} steps ({num_timesteps/self.fps:.1f}s). "
              f"Peaks -> Goal: {np.max(goal_probs):.3f}, ShotsOn: {np.max(shot_on_probs):.3f}, "
              f"ShotsOff: {np.max(shot_off_probs):.3f}, Penalty: {np.max(penalty_probs):.3f}, "
              f"Yellow: {np.max(yellow_probs):.3f}, Red: {np.max(red_probs):.3f}", flush=True)

        # Multi-signal highlight candidate generation
        # Goal detection:
        # A goal in soccer highlight videos requires either:
        # 1) Direct strong Goal peak (>= 0.12), OR
        # 2) Definite Goal elevation (>= 0.075) within +/- 6 seconds of a Shot on Target (>= 0.15)
        candidates = []
        radius_steps = int(6.0 * self.fps)  # 6 seconds correlation window

        # 1. Spot Goal candidates
        goal_peak_indices = np.where(goal_probs >= 0.15)[0]
        for idx in goal_peak_indices:
            g_conf = float(goal_probs[idx])
            start_w = max(0, idx - radius_steps)
            end_w = min(num_timesteps, idx + radius_steps + 1)
            max_shot_nearby = float(np.max(shot_on_probs[start_w:end_w])) if end_w > start_w else 0.0
            
            is_valid_goal = (g_conf >= 0.20) or (g_conf >= 0.15 and max_shot_nearby >= 0.20)
            if is_valid_goal:
                # Combined confidence reflecting both the shot and goal signal
                combined_conf = round(max(g_conf, max_shot_nearby * 0.9), 3)
                candidates.append({
                    "timestep": idx,
                    "event_type": "GOAL",
                    "confidence": combined_conf,
                    "priority": 10,
                    "raw_class": "Goal",
                    "raw_class_id": 2
                })

        # 2. Spot Penalty candidates
        penalty_indices = np.where(penalty_probs >= 0.22)[0]
        for idx in penalty_indices:
            candidates.append({
                "timestep": idx,
                "event_type": "PENALTY",
                "confidence": round(float(penalty_probs[idx]), 3),
                "priority": 8,
                "raw_class": "Penalty",
                "raw_class_id": 0
            })

        # 3. Spot Dangerous Attacks / Shots on target
        shot_on_indices = np.where(shot_on_probs >= 0.22)[0]
        for idx in shot_on_indices:
            candidates.append({
                "timestep": idx,
                "event_type": "ATTACK",
                "confidence": round(float(shot_on_probs[idx]), 3),
                "priority": 5,
                "raw_class": "Shots on target",
                "raw_class_id": 5
            })

        # 4. Spot Close Misses / Shots off target
        shot_off_indices = np.where(shot_off_probs >= 0.28)[0]
        for idx in shot_off_indices:
            candidates.append({
                "timestep": idx,
                "event_type": "ATTACK",
                "confidence": round(float(shot_off_probs[idx]), 3),
                "priority": 4,
                "raw_class": "Shots off target",
                "raw_class_id": 6
            })

        # 5. Spot Cards (Yellow / Red)
        card_indices = np.where((yellow_probs >= 0.22) | (red_probs >= 0.22))[0]
        for idx in card_indices:
            is_red = red_probs[idx] >= yellow_probs[idx]
            conf = float(red_probs[idx] if is_red else yellow_probs[idx])
            candidates.append({
                "timestep": idx,
                "event_type": "CARD",
                "confidence": round(conf, 3),
                "priority": 3,
                "raw_class": "Red card" if is_red else "Yellow card",
                "raw_class_id": 15 if is_red else 14
            })

        if not candidates:
            print("[Spivak] No candidate events met the criteria.", flush=True)
            return []

        # Sort candidates by:
        # 1) Priority (GOAL > PENALTY > ATTACK > CARD)
        # 2) Confidence descending
        candidates = sorted(candidates, key=lambda x: (x["priority"], x["confidence"]), reverse=True)

        # Per-class Non-Maximum Suppression.
        #
        # Each class tracks its own last-accepted timestamp independently.
        # Events of DIFFERENT classes within the cooldown window are NOT suppressed
        # against each other — e.g. a foul (ATTACK) immediately followed by a yellow
        # card (CARD) correctly produces two separate highlight events.
        #
        # Cooldown values:
        #   GOAL/PENALTY: 20s — long window to de-duplicate broadcast replays
        #   CARD:         15s — referee usually shows the card once
        #   ATTACK:        8s — clustered shots should still be deduplicated
        CLASS_COOLDOWNS_STEPS = {
            "GOAL":    int(20.0 * self.fps),
            "PENALTY": int(20.0 * self.fps),
            "ATTACK":  int(8.0 * self.fps),
            "CARD":    int(15.0 * self.fps),
        }
        last_accepted_step = {k: -99999 for k in CLASS_COOLDOWNS_STEPS}
        selected_events = []

        for cand in candidates:
            idx = cand["timestep"]
            etype = cand["event_type"]
            cooldown = CLASS_COOLDOWNS_STEPS.get(etype, int(15.0 * self.fps))

            # Skip if too close to last accepted event of the SAME class
            if idx - last_accepted_step[etype] < cooldown:
                continue

            last_accepted_step[etype] = idx
            timestamp = idx / self.fps
            selected_events.append({
                "timestamp": round(timestamp, 2),
                "minute": int(timestamp // 60),
                "event_type": etype,
                "confidence": cand["confidence"],
                "raw_class": cand["raw_class"],
                "raw_class_id": cand["raw_class_id"]
            })

        # Sort final events chronologically
        selected_events = sorted(selected_events, key=lambda x: x["timestamp"])
        for ev in selected_events:
            print(f"[Spivak] Verified highlight: {ev['event_type']} ({ev['raw_class']}) at {ev['timestamp']}s (min {ev['minute']}) conf={ev['confidence']}", flush=True)

        return selected_events

def extract_soccer_events_spivak(video_path, min_confidence=0.25, progress_callback=None):
    """
    Main entry point for Spivak video action spotting:
    1. Extract ResNet-152 features (at 2 FPS)
    2. Temporal U-Net inference & NMS
    3. Return events list and video duration
    """
    spotter = SpivakSoccerSpotter.get_instance()
    
    if progress_callback:
        progress_callback(5, "Memulai ekstraksi fitur video (ResNet-152 @ 2.0 FPS)...")

    features, duration = spotter.extract_features(
        video_path,
        batch_size=32,
        progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback(80, "Menganalisis temporal anchor Spivak...")

    events = spotter.spot_actions(
        features,
        min_confidence=min_confidence,
        stride=56,
        progress_callback=progress_callback
    )

    return events, duration
