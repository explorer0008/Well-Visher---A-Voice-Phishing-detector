import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
SAMPLE_RATE = 16000  # Whisper works best with 16kHz
def record_audio(duration: int = 15, filename: str = "recorded_audio.wav") -> str:
    
    print(f"  🔴 Recording for {duration} seconds... Speak now!")

    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()

    output_path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    wav.write(output_path, SAMPLE_RATE, audio_data)

    print(f"  ✅ Audio saved to {output_path}")
    return output_path
