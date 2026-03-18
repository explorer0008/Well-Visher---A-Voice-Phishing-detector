# Vishing Detection System

Real-time voice phishing (vishing) detector using a hybrid ZSL + BERT pipeline.

## Setup

```bash
pip install -r requirements.txt
```

### WhisperX (optional but recommended)
```bash
pip install whisperx
# requires ffmpeg: https://ffmpeg.org/download.html
```

### Vosk (optional)
```bash
pip install vosk
# Download English model (≈1.8 GB):
# https://alphacephei.com/vosk/models  → vosk-model-en-us-0.22
# Extract it to the project root so the path is:
#   vishing_detector/vosk-model-en-us-0.22/
```

## Run

```bash
python main.py
```

## Step 3 — Accuracy Comparison

After recording, type the exact words you spoke into the reference box and
click **Compare All Engines**. The system will:

1. Re-transcribe the audio with **WhisperX** and **Vosk** in parallel.
2. Compute Word Error Rate (WER) for each engine against your reference text.
3. Show per-engine accuracy, error breakdown (substitutions / deletions / insertions).
4. Compute **cross-engine similarity** — high similarity across all three engines
   indicates the transcription is reliable and can be trusted for classification.
5. Show a **Consensus Accuracy** (average of all available engines).

## Project Structure

```
vishing_detector/
├── main.py                        # GUI entry point
├── demo.py                        # CLI demo with sample texts
├── requirements.txt
├── models/
│   └── bert_classifier.py         # Hybrid ZSL+BERT vishing classifier
├── utils/
│   ├── audio_recorder.py          # Microphone recording helper
│   ├── transcriber.py             # Faster-Whisper, WhisperX, Vosk wrappers
│   ├── accuracy_comparator.py     # WER + cross-engine similarity
│   └── zsl_labeler.py             # Zero-shot classification labels
└── data/                          # Recorded WAV files (auto-created)
```
