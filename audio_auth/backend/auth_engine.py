
import torch
import torchaudio
import os
import shutil

# Monkeypatch os.symlink for Windows to use copy (avoids Admin Requirement)
if os.name == "nt":
    def _symlink(src, dst, target_is_directory=False):
        if target_is_directory:
             if os.path.exists(dst):
                shutil.rmtree(dst)
             shutil.copytree(src, dst)
        else:
             if os.path.exists(dst):
                 os.remove(dst)
             shutil.copy2(src, dst)
    # We only patch if we can't symlink? Or just always force it?
    # SpeechBrain seems to force symlink.
    os.symlink = _symlink

# Monkeypatch torchaudio.list_audio_backends if missing (removed in torchaudio 2.1+)
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile", "ffmpeg"]
    torchaudio.list_audio_backends = _list_audio_backends

from speechbrain.inference.speaker import SpeakerRecognition
import whisper
import numpy as np
from difflib import SequenceMatcher
import os

class SpeakerRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SpeakerRecognition model on {self.device}...")
        # Spkrec-ecapa-voxceleb is a strong model for speaker verification
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model",
            run_opts={"device": self.device}
        )

    def get_embedding(self, audio_path):
        """
        Computes the speaker embedding for a given audio file.
        Returns a numpy array.
        """
        # Explicitly use soundfile to avoid Torchaudio backend issues
        # signal, fs = torchaudio.load(audio_path, backend="soundfile")
        import soundfile as sf
        signal_np, fs = sf.read(audio_path)
        # Convert to tensor
        signal = torch.from_numpy(signal_np).float()
        # If mono, soundfile returns 1D array. Torchaudio load returns (channels, time) usually.
        # SpeechBrain expects (batch, time) or (time).
        # If we have (time), encode_batch handles it.
        # But wait, Torchaudio.load returns (C, T). Soundfile returns (T, C) or (T,).
        
        # Ensure we have consistent dimensions.
        if len(signal.shape) == 1:
             # (T) -> (1, T) to mimic torchaudio (C, T) if needed, but SpeechBrain might want just (T)
             # Actually, let's verify what SpeechBrain expects. 
             # encode_batch expects (batch, time).
             # If we pass (T), it might work if we unsqueeze.
             pass
        elif len(signal.shape) == 2:
            # (T, C) -> (C, T)
             signal = signal.transpose(0, 1)
        
        # But wait, speechbrain model.encode_batch takes wavs: torch.Tensor
        # Wavs is expected to be (Batch, Time).
        
        # If I have a single file, I should probably unsqueeze(0) to make it batch size 1.
        if signal.ndim == 1:
             signal = signal.unsqueeze(0) # (1, T) for Batch=1
        elif signal.ndim == 2:
             # If (C, T), and C=1, then it is (1, T).
             # If C > 1, take first channel?
             # My ffmpeg command forces -ac 1 so it should be mono.
             if signal.shape[0] == 1:
                 pass # Already (1, T)
             else:
                 # Unexpected channel dim, maybe (T, C) from soundfile?
                 pass 

        # Let's simplify:
        # FFMPEG guarantees Mono. Soundfile reads as (Time,).
        # We need (Batch=1, Time).
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        elif signal.ndim == 2:
             # If soundfile returned (Time, Channels), transpose to (Channels, Time)
             if signal.shape[0] > signal.shape[1]: 
                 signal = signal.transpose(0, 1)
             # Now (C, T). If C=1, ok.
             pass
        # The model expects a batch dimension, but encode_batch handles it if we pass a tensor
        # However, let's be safe and ensure the signal is correct
        embedding = self.model.encode_batch(signal)
        # Squeeze to remove batch dimension and convert to numpy
        return embedding.squeeze().cpu().detach().numpy()

    def verify_speaker(self, audio_path, enrolled_embedding_path, threshold=0.25):
        """
        Verifies if the speaker in audio_path matches the enrolled embedding.
        Returns tuple (is_match, score)
        """
        # Compute embedding for the new audio
        new_embedding = self.get_embedding(audio_path)
        new_embedding_tensor = torch.from_numpy(new_embedding).to(self.model.device)
        
        # Load enrolled embedding
        if not os.path.exists(enrolled_embedding_path):
            raise FileNotFoundError(f"Enrolled embedding not found at {enrolled_embedding_path}")
            
        enrolled_embedding = np.load(enrolled_embedding_path)
        enrolled_embedding_tensor = torch.from_numpy(enrolled_embedding).to(self.model.device)
        
        # Compute Cosine Similarity
        # We need to add dimensions back for cosine_similarity to work properly if they are 1D
        if new_embedding_tensor.ndim == 1:
            new_embedding_tensor = new_embedding_tensor.unsqueeze(0)
        if enrolled_embedding_tensor.ndim == 1:
            enrolled_embedding_tensor = enrolled_embedding_tensor.unsqueeze(0)
            
        score = torch.nn.functional.cosine_similarity(new_embedding_tensor, enrolled_embedding_tensor)
        score_val = score.item()
        
        # Threshold for ECAPA-VoxCeleb is typically around 0.25-0.35 for verification
        return score_val > threshold, score_val

class ContentVerifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model on {self.device}...")
        self.model = whisper.load_model("tiny", device=self.device)

    def verify_content(self, audio_path, expected_text, threshold=0.8):
        """
        Transcribes audio and checks if it matches expected text.
        Returns tuple (is_match, similarity_score, transcribed_text)
        """
        result = self.model.transcribe(audio_path)
        transcribed_text = result["text"].strip().lower()
        
        # Normalize expected text
        expected_text = expected_text.strip().lower()
        
        # Calculate similarity
        matcher = SequenceMatcher(None, transcribed_text, expected_text)
        similarity = matcher.ratio()
        
        return similarity >= threshold, similarity, transcribed_text
