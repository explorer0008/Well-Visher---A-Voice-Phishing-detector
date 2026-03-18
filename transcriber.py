"""
Transcriber - Multi-engine speech-to-text
Engines: Faster-Whisper (primary), WhisperX, Vosk
"""


import os
os.environ["PATH"] += r";C:\Users\sharm\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
MODEL_SIZE = "base"
_whisper_model = None


# ── Faster-Whisper ────────────────────────────────────────────────
def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"  Loading Faster-Whisper model ({MODEL_SIZE})...")
        _whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("  Faster-Whisper loaded ✅")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Primary transcription using Faster-Whisper."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = _get_whisper_model()
    segments, _ = model.transcribe(
        audio_path, beam_size=5, language="en",
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
    )
    transcript = " ".join(seg.text.strip() for seg in segments)
    return transcript if transcript else "[No speech detected]"


# ── WhisperX ──────────────────────────────────────────────────────
def transcribe_whisperx(audio_path: str) -> str:
    """Transcription using WhisperX."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        import whisperx
        model = whisperx.load_model(MODEL_SIZE, device="cpu", compute_type="int8")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8, language="en")
        segments = result.get("segments", [])
        transcript = " ".join(seg["text"].strip() for seg in segments)
        return transcript if transcript else "[No speech detected]"
    except ImportError:
        return "[WhisperX not installed — run: pip install whisperx]"
    except Exception as e:
        return f"[WhisperX error: {str(e)}]"


# ── Vosk ──────────────────────────────────────────────────────────
def transcribe_vosk(audio_path: str, model_path: str = "vosk-model-en-us-0.22") -> str:
    """Transcription using Vosk (offline, fast)."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    try:
        import wave, json
        from vosk import Model, KaldiRecognizer

        if not os.path.exists(model_path):
            return (
                f"[Vosk model not found at '{model_path}'. "
                "Download from https://alphacephei.com/vosk/models "
                "and extract to project root.]"
            )

        wf = wave.open(audio_path, "rb")
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            return "[Vosk requires mono 16-bit WAV. Re-record audio.]"

        vosk_model = Model(model_path)
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                r = json.loads(rec.Result())
                if r.get("text"):
                    results.append(r["text"])
        final = json.loads(rec.FinalResult())
        if final.get("text"):
            results.append(final["text"])

        transcript = " ".join(results).strip()
        return transcript if transcript else "[No speech detected]"
    except ImportError:
        return "[Vosk not installed — run: pip install vosk]"
    except Exception as e:
        return f"[Vosk error: {str(e)}]"
