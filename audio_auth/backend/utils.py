
import subprocess
import shutil
import os

def is_ffmpeg_installed():
    return shutil.which("ffmpeg") is not None

def convert_webm_to_wav(input_path, output_path):
    """
    Converts a WebM audio file to a WAV file with 16kHz sampling rate and mono channel.
    This is required for SpeechBrain models.
    """
    if not is_ffmpeg_installed():
        raise RuntimeError("FFmpeg is not installed. Please install FFmpeg to process audio.")
    
    # -ac 1: Mono
    # -ar 16000: 16kHz sample rate
    command = [
        "ffmpeg", "-y", "-i", input_path, 
        "-ac", "1", "-ar", "16000", output_path
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise e
