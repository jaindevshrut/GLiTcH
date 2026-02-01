
import torchaudio
import soundfile
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Soundfile version: {soundfile.__version__}")

try:
    print("Available backends:", torchaudio.list_audio_backends())
except Exception as e:
    print("Error listing backends:", e)

try:
    # Try setting backend explicitly
    torchaudio.set_audio_backend("soundfile")
    print("Set backend to soundfile")
except Exception as e:
    print("Error setting backend:", e)
    
print("Current backend:", torchaudio.get_audio_backend())
