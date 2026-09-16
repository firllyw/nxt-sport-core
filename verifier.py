"""
verifier.py — Local event verification via audio signal analysis.

Provides referee whistle detection using a bandpass filter (2–4 kHz).
Used as a secondary confirmation signal after Spivak temporal detection.

Design decisions:
- Crowd cheer detection is intentionally EXCLUDED. Local match footage without
  a stadium crowd does not produce reliable amplitude spikes on goals.
- Whistle detection is used ONLY for PENALTY and CARD events.
  It is intentionally NOT applied to GOAL events because fouls, kickoffs,
  and corners also produce referee whistles — making the signal ambiguous
  for goal confirmation. Goal accuracy relies on elevated detection thresholds
  in spivak_service.py (Phase A) and feature extractor quality (Phase D).
"""

import os
import numpy as np


def detect_whistle(audio_wav_path, threshold_db=-32.0):
    """
    Detect referee whistle presence in a WAV audio file.

    Strategy:
    1. Load audio (mono, any sample rate)
    2. Apply bandpass filter (2000–4000 Hz) — the referee whistle frequency range
    3. Compute short-time RMS energy on filtered signal
    4. Return True if peak energy exceeds threshold_db

    Args:
        audio_wav_path: Absolute path to a WAV file, or None
        threshold_db:   Energy threshold in dB. Tune this lower (-40) to be more
                        sensitive, or higher (-25) to require a louder whistle.

    Returns:
        bool: True if a whistle-like burst was detected
    """
    if audio_wav_path is None or not os.path.exists(audio_wav_path):
        return False

    if os.path.getsize(audio_wav_path) < 1024:
        # File too small to be meaningful audio
        return False

    try:
        import librosa
        from scipy.signal import butter, sosfilt

        y, sr = librosa.load(audio_wav_path, sr=None, mono=True)

        if len(y) < sr * 0.5:
            # Less than 0.5 seconds — not enough signal
            return False

        nyq = sr / 2.0
        low_hz = 2000.0
        high_hz = 4000.0

        if nyq <= low_hz:
            # Sample rate too low to capture whistle frequencies (< 8 kHz audio)
            print(f"[Verifier] Warning: sample rate {sr} Hz too low for whistle detection.", flush=True)
            return False

        low = low_hz / nyq
        high = min(high_hz / nyq, 0.99)  # clamp to avoid numerical issues

        if low >= high:
            return False

        sos = butter(4, [low, high], btype='bandpass', output='sos')
        filtered = sosfilt(sos, y)

        # Short-time RMS energy (25ms frames, 10ms hop)
        frame_length = max(64, int(sr * 0.025))
        hop_length = max(32, int(sr * 0.010))
        rms = librosa.feature.rms(
            y=filtered,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        if len(rms) == 0 or np.max(rms) < 1e-10:
            return False

        peak_db = 20.0 * np.log10(float(np.max(rms)) + 1e-10)
        detected = peak_db > threshold_db

        print(
            f"[Verifier] Whistle check: peak={peak_db:.1f} dB, "
            f"threshold={threshold_db} dB, detected={detected}",
            flush=True
        )
        return detected

    except ImportError:
        print("[Verifier] librosa/scipy not installed — whistle detection skipped.", flush=True)
        return False
    except Exception as exc:
        print(f"[Verifier] Whistle detection error: {exc}", flush=True)
        return False


def verify_event(audio_wav_path, event_type, spivak_confidence, mode="full"):
    """
    Decide whether a Spivak-proposed event should be accepted as a real highlight.

    Verification logic per event type:

        GOAL    — No audio gate.
                  Fouls, kickoffs, corners all produce whistles, so whistle presence
                  cannot discriminate goals. Rely on the elevated detection threshold
                  set in spivak_service.py (Phase A) as the primary quality gate.
                  Result: always accepted here (gate is upstream).

        PENALTY — Whistle IS expected (always a foul in the penalty box).
                  Accept if whistle detected OR confidence >= 0.22.

        CARD    — Whistle IS expected (a foul always precedes a card).
                  Accept if whistle detected OR confidence >= 0.22.

        ATTACK  — No audio gate. Shots do not reliably stop play.
                  Phase A NMS threshold is the sole quality gate.
                  Result: always accepted here.

    Args:
        audio_wav_path:    Path to WAV file extracted from candidate clip, or None
        event_type:        One of "GOAL", "PENALTY", "CARD", "ATTACK"
        spivak_confidence: Raw confidence value from Spivak model output
        mode:              "full"         — run full verification
                           "whistle_only" — only run whistle check, skip threshold gate
                           "disabled"     — bypass all verification (accept everything)

    Returns:
        bool: True if the event should be kept as a highlight
    """
    if mode == "disabled":
        return True

    etype = event_type.upper()

    if etype == "GOAL":
        # Intentionally no audio gate.
        # The detection threshold in spivak_service.py is the primary filter.
        print(
            f"[Verifier] GOAL at confidence={spivak_confidence:.3f} → accepted "
            f"(audio gate intentionally disabled for goals — fouls also whistle)",
            flush=True
        )
        return True

    elif etype == "PENALTY":
        whistle = detect_whistle(audio_wav_path)
        accepted = whistle or spivak_confidence >= 0.22
        print(
            f"[Verifier] PENALTY: whistle={whistle}, "
            f"confidence={spivak_confidence:.3f} → accepted={accepted}",
            flush=True
        )
        return accepted

    elif etype == "CARD":
        whistle = detect_whistle(audio_wav_path)
        accepted = whistle or spivak_confidence >= 0.22
        print(
            f"[Verifier] CARD: whistle={whistle}, "
            f"confidence={spivak_confidence:.3f} → accepted={accepted}",
            flush=True
        )
        return accepted

    elif etype == "ATTACK":
        # Shots do not reliably trigger a whistle.
        # Phase A NMS threshold is the only gate for attacks.
        print(
            f"[Verifier] ATTACK at confidence={spivak_confidence:.3f} → accepted "
            f"(no audio gate for attacks)",
            flush=True
        )
        return True

    else:
        # Unknown event type — accept by default to avoid silent data loss
        print(f"[Verifier] Unknown event type '{etype}' → accepted by default", flush=True)
        return True
