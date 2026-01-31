import gradio as gr
import random
import time
import base64
import tempfile
import os

def detect_ai_audio(audio):
    """
    Dummy AI audio detection function.
    Returns 0 (human) or 1 (AI) randomly for testing purposes.
    
    In production, replace this with your actual model inference:
    - Load your trained model
    - Extract audio features (MFCC, spectrograms, etc.)
    - Run inference
    - Return 0 for human, 1 for AI
    """
    
    # Handle different input types
    audio_file = None
    
    if audio is None:
        # No audio received
        return random.choice([0, 1])
    
    # If it's a base64 string (from the extension)
    if isinstance(audio, str):
        if audio.startswith('data:'):
            # Extract base64 data after the comma
            try:
                base64_data = audio.split(',')[1]
                audio_bytes = base64.b64decode(base64_data)
                
                # Save to temp file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as f:
                    f.write(audio_bytes)
                    audio_file = f.name
            except Exception as e:
                print(f"Error decoding base64: {e}")
        else:
            audio_file = audio
    elif isinstance(audio, tuple):
        # Gradio audio format: (sample_rate, audio_array)
        audio_file = "direct_input"
    else:
        audio_file = audio
    
    # Simulate processing time (like a real model would take)
    time.sleep(0.5 + random.random() * 0.5)
    
    # Clean up temp file if created
    if audio_file and audio_file != "direct_input" and os.path.exists(audio_file):
        try:
            # In real implementation, you would process the audio here
            # For demo, we just delete the temp file
            os.unlink(audio_file)
        except:
            pass
    
    # Random result for testing: 0 = Human, 1 = AI
    # In production, replace with actual model prediction
    result = random.choice([0, 1])
    
    print(f"Processed audio, result: {result} ({'AI' if result == 1 else 'Human'})")
    
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=detect_ai_audio,
    inputs=gr.Audio(type="filepath", label="Audio Input"),
    outputs=gr.Number(label="Detection Result (0=Human, 1=AI)"),
    title="AI Voice Detector (Demo)",
    description="""
    Upload audio to detect if it's AI-generated. 
    
    **Results:**
    - **0** = Human Voice
    - **1** = AI Generated Voice
    
    *Note: This is a demo that returns random results. 
    Replace with your actual model for production use.*
    """,
    examples=None,
    allow_flagging="never",
    api_name="predict"
)

if __name__ == "__main__":
    demo.launch()

