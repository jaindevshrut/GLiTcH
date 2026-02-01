
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
import uuid
import random
from pydantic import BaseModel
from typing import Optional
import numpy as np

# Monkeypatch torchaudio
import torch
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile", "ffmpeg"]
    torchaudio.list_audio_backends = _list_audio_backends

from utils import convert_webm_to_wav
# auth_engine already imports correctly but if we import SpeakerRecognizer from auth_engine, it should be fine.
from auth_engine import SpeakerRecognizer, ContentVerifier

app = FastAPI()

# CORSMiddleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = "uploads"
EMBEDDINGS_DIR = "embeddings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# User Database
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Load Models (Global to avoid reloading)
print("Initializing Models...")
# We initialize them lazily or here. Here is fine.
# Note: This might take time on startup.
speaker_recognizer = SpeakerRecognizer()
content_verifier = ContentVerifier()
print("Models Initialized.")

CHALLENGE_PHRASES = [
    "The quick brown fox jumps over the lazy dog",
    "Authentication is secure with voice",
    "Open the pod bay doors HAL",
    "My voice is my password verify me",
    "Artificial intelligence is transforming the world"
]

@app.get("/api/challenge")
def get_challenge():
    return {"challenge": random.choice(CHALLENGE_PHRASES)}

@app.post("/api/register")
async def register(username: str = Form(...), audio: UploadFile = File(...)):
    users = load_users()
    if username in users:
        raise HTTPException(status_code=400, detail="User already exists")
    
    print(f"Registering user: {username}")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    
    try:
        with open(webm_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        print(f"Saved webm to {webm_path}")
        
        # Convert to WAV
        convert_webm_to_wav(webm_path, wav_path)
        print(f"Converted to wav at {wav_path}")
            
        # Generate Embedding
        embedding = speaker_recognizer.get_embedding(wav_path)
        embedding_filename = f"{username}_embedding.npy"
        embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)
        np.save(embedding_path, embedding)
        print(f"Saved embedding to {embedding_path}")
        
        # Save User (store relative path or absolute? store relative to backend for portability)
        # But we are running from backend dir usually.
        users[username] = {
            "embedding_path": embedding_path
        }
        save_users(users)
        
        return {"message": "User registered successfully"}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Registration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(webm_path): os.remove(webm_path)
        if os.path.exists(wav_path): os.remove(wav_path)

@app.post("/api/login")
async def login(username: str = Form(...), challenge: str = Form(...), audio: UploadFile = File(...)):
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
        
    print(f"Login attempt for {username} with challenge '{challenge}'")
        
    user_data = users[username]
    enrolled_embedding_path = user_data["embedding_path"]
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(UPLOAD_DIR, f"{file_id}.webm")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    
    try:
        with open(webm_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        convert_webm_to_wav(webm_path, wav_path)
        
        # 1. Verify Content (Liveness)
        is_content_match, content_score, transcribed = content_verifier.verify_content(wav_path, challenge)
        print(f"Content Verification: Match={is_content_match}, Score={content_score:.2f}, Transcribed='{transcribed}'")
        
        if not is_content_match:
            raise HTTPException(status_code=401, detail=f"Content mismatch. You said: '{transcribed}', Expected: '{challenge}'")

        # 2. Verify Speaker
        is_speaker_match, speaker_score = speaker_recognizer.verify_speaker(wav_path, enrolled_embedding_path)
        print(f"Speaker Verification: Match={is_speaker_match}, Score={speaker_score:.2f}")
        
        if is_speaker_match and speaker_score >= 0.60:
            return {
                "status": "success", 
                "message": "Login successful", 
                "speaker_score": speaker_score,
                "content_score": content_score
            }
        else:
            raise HTTPException(status_code=401, detail=f"Voice mismatch. Verification Score: {speaker_score:.2f}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Login Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(webm_path): os.remove(webm_path)
        if os.path.exists(wav_path): os.remove(wav_path)

# Mount frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
