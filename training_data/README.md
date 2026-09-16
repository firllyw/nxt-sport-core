# Training Data — Feature Extractor Fine-Tuning

This directory holds labeled video clips used to fine-tune the ResNet-152 feature
extractor so it produces features closer to the SoccerNet distribution expected by
the Spivak model.

---

## Directory Structure

```
training_data/
  goal/         ← clips of goal moments (label: 1)
  no_event/     ← clips of regular play (label: 0)
```

---

## How Many Clips Do I Need?

| Class | Minimum | Recommended |
|---|---|---|
| `goal/` | 20 clips | 40–60 clips |
| `no_event/` | 30 clips | 80–120 clips |

More is better, but even 20/30 will produce a meaningful improvement over generic ImageNet weights.

---

## Step 1 — Download YouTube Soccer Clips

Use `yt-dlp` to download at moderate resolution (480p is plenty):

```bash
# Install yt-dlp if needed
pip install yt-dlp

# Download a YouTube video (e.g. "Top 50 Goals" compilation)
yt-dlp -f "best[height<=480]" -o "full_compilation.%(ext)s" "https://youtube.com/watch?v=..."
```

Good search terms:
- "top 50 goals Premier League 2023"
- "best goals La Liga 2024 compilation"
- "top goals Champions League highlights"

---

## Step 2 — Trim Individual Goal Moments

For each goal in the compilation, trim a ~8 second window **centered on the moment the ball enters the net**. The exact center doesn't need to be perfect — ±2s is fine.

```bash
# Format: ffmpeg -ss <start> -i <input> -t <duration> -c copy <output>
# Example: goal at 01:23 in the video, trim 4s before → 4s after

ffmpeg -ss 00:01:19 -i full_compilation.mp4 -t 8 -c copy training_data/goal/goal_001.mp4
ffmpeg -ss 00:03:41 -i full_compilation.mp4 -t 8 -c copy training_data/goal/goal_002.mp4
# ... repeat for each goal moment
```

---

## Step 3 — Collect No-Event Clips

For `no_event/`, clip segments of regular play from any soccer video. 
These should NOT contain a goal moment. Variety is important:

- Midfield passing sequences
- Goal kicks / throw-ins
- Tactical dead balls (no action)
- Defensive clearances

```bash
# Example: clip 3 random background segments from a match video
ffmpeg -ss 00:05:00 -i match.mp4 -t 8 -c copy training_data/no_event/bg_001.mp4
ffmpeg -ss 00:22:30 -i match.mp4 -t 8 -c copy training_data/no_event/bg_002.mp4
ffmpeg -ss 00:55:10 -i match.mp4 -t 8 -c copy training_data/no_event/bg_003.mp4
```

---

## Step 4 — Run Fine-Tuning

```bash
cd nxt-sport-core

python train_feature_extractor.py \
  --training-dir ./training_data \
  --output-weights ./models/feature_extractor/resnet_finetuned.h5 \
  --epochs 10 \
  --batch-size 8
```

**Expected runtime (CPU, no GPU):** 2–4 hours for ~100 clips.

The script will:
1. Sample 20 frames from each clip
2. Fine-tune only the last ResNet-152 residual block (`conv5_block3`)
3. Save backbone weights to `models/feature_extractor/resnet_finetuned.h5`
4. Update `models/spivak_soccernet/resnet_normalizer.pkl` to match new feature distribution

---

## Step 5 — Restart the Server

After fine-tuning, restart `nxt-sport-core`. The new weights are loaded automatically at startup:

```
[Extractor] Fine-tuned weights loaded from: models/feature_extractor/resnet_finetuned.h5
```

---

## Long-Term Improvement (Hop 2)

YouTube clips close the gap between *ImageNet* and *SoccerNet broadcast*.
They do NOT close the gap between *SoccerNet broadcast* and *your local camera*.

Once you have processed some matches, manually identify the true goal timestamps
and export those clips into a second fine-tune dataset:

```
training_data_own/
  goal/       ← clips from YOUR matches
  no_event/   ← background from YOUR matches
```

Then run fine-tuning again, loading the YouTube-tuned weights as the starting point:

```bash
# Copy the YouTube-tuned weights as the new "imagenet" starting point
cp models/feature_extractor/resnet_finetuned.h5 models/feature_extractor/resnet_youtube.h5

python train_feature_extractor.py \
  --training-dir ./training_data_own \
  --output-weights ./models/feature_extractor/resnet_finetuned.h5 \
  --epochs 10 --batch-size 8
```

This two-hop approach progressively narrows the domain gap without requiring a
large dataset at either step.
