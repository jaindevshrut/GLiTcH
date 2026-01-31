# AI Audio Detector - Chrome Extension

Real-time detection of AI-generated audio during video calls, meetings, and interviews.

## 🚀 Quick Start (Demo Mode - No API Needed!)

1. **Load the Extension**:
   - Open Chrome → `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked" → Select this folder

2. **Test Immediately**:
   - Click the extension icon
   - Toggle **"Demo Mode"** ON ✅
   - Click **"Start Monitoring"**
   - Every 10 seconds → Random AI/Human result!

## Features

| Feature | Description |
|---------|-------------|
| 🎤 **Microphone Mode** | Capture audio from your mic |
| 🔊 **Video Call Mode** | Capture the OTHER person's audio in video calls |
| ⏱️ **10-Second Clips** | Analyzes audio in 10-second chunks |
| 💾 **Local Storage** | Saves clips to IndexedDB (no corruption) |
| 🎭 **Demo Mode** | Test without API - random 0/1 results |
| 📊 **History Log** | Track all detection results |
| ⬇️ **Download Clips** | Save audio clips for debugging |

## How Demo Mode Works

```
┌─────────────────────────────────────────┐
│  10 seconds of audio recorded           │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Stored locally (IndexedDB)             │
│  No corruption - clean blobs            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Demo: Random 0 or 1                    │
│  Real: Send to HuggingFace → Get 0/1    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  0 = ✅ Human Voice                     │
│  1 = 🤖 AI Detected                     │
└─────────────────────────────────────────┘
```

## Audio Sources

### 🎤 Microphone
- Captures YOUR microphone
- Good for testing

### 🔊 Video Call (Tab Audio)
- Captures audio from the browser tab
- Use during video calls to analyze the **other person's voice**
- **Works with**: Google Meet, Zoom (web), Teams, Webex
- ⚠️ Must be on the video call tab when clicking "Start"

## Full Setup (With HuggingFace API)

### 1. Deploy the HuggingFace Space

```bash
# Go to huggingface.co/spaces
# Create new Space with Gradio SDK
# Upload files from huggingface-space/ folder:
#   - app.py
#   - requirements.txt
```

### 2. Configure Extension

1. Disable "Demo Mode"
2. Enter your Space URL: `https://your-username-your-space.hf.space`
3. Click 💾 to save
4. Start monitoring!

## Output Format

| Result | Meaning | Display |
|--------|---------|---------|
| `0` | Human Voice | ✅ Green indicator |
| `1` | AI Generated | 🤖 Red warning |

## Integrating Your Real Model

Replace the dummy function in `huggingface-space/app.py`:

```python
def detect_ai_audio(audio):
    # 1. Load your trained deepfake detection model
    model = load_model("your_deepfake_detector.h5")
    
    # 2. Extract audio features
    features = extract_mfcc(audio)  # or spectrograms, etc.
    
    # 3. Run inference
    prediction = model.predict(features)
    
    # 4. Return 0 (human) or 1 (AI)
    return 1 if prediction > 0.5 else 0
```

## Files Structure

```
glitch-extension/
├── manifest.json       # Extension permissions
├── popup.html          # UI with Demo Mode toggle
├── popup.css           # Styling
├── popup.js            # Recording + API logic
├── icons/              # Extension icons
└── huggingface-space/
    ├── app.py          # Gradio API (dummy → replace with model)
    ├── requirements.txt
    └── README.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | Allow microphone in browser settings |
| Tab capture failed | Make sure you're on a video call page |
| API errors | Check HuggingFace Space is running, or use Demo Mode |
| No results | Check if clip is recording (see timer) |

## Privacy

- **Demo Mode**: 100% local, no data sent anywhere
- **API Mode**: Audio only sent to YOUR HuggingFace Space
- **Local Storage**: Clips stored in browser's IndexedDB
- **No tracking**: Extension doesn't collect any data

## License

MIT
