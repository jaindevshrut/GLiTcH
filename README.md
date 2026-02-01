# 🎭 GLiTcH - AI Voice Deepfake Detector

<div align="center">


**Real-time AI-generated voice detection for video calls & audio streams**

[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://chrome.google.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Hosted-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

---

## 🎯 What is GLiTcH?

GLiTcH is a Chrome extension that detects **AI-generated/deepfake voices** in real-time during video calls, live streams, or any audio playing in your browser. It helps you identify if the person you're talking to is using a synthetic voice.

### 🔥 Key Features

- 🎙️ **Real-time Detection** - Analyzes audio while you listen
- 🌐 **Works Everywhere** - YouTube, Google Meet, Zoom, Teams, and more
- 🔊 **Non-intrusive** - Audio keeps playing while analyzing
- ⚡ **Fast Results** - Get detection results in ~15 seconds
- 🎨 **Visual Indicators** - Clear badge showing AI or Human

---

## 🧠 The Models Behind GLiTcH

GLiTcH uses an ensemble of four specialized deep learning models, each trained on different audio feature representations:

### 1️⃣ best_mel_cnn.pt (Primary Model) ⭐
```
📊 Architecture: CNN trained on Mel-scale spectrograms
🎯 Specialty: Best overall deepfake detection performance
⚡ Performance: ACC 92.76% | EER 4.10%
🔧 Threshold: 0.521
```

### 2️⃣ best_lfcc (LFCC-based Model)
```
📊 Architecture: CNN trained on Linear Frequency Cepstral Coefficients
🎯 Specialty: Captures fine-grained spectral characteristics
⚡ Performance: ACC 90.55% | EER 4.96%
🔧 Threshold: 0.618
```

### 3️⃣ best_mel (Mel Spectrogram Model) ⭐ Best Overall
```
📊 Architecture: CNN trained on Mel-scale spectrograms
🎯 Specialty: Human auditory perception-aligned features
⚡ Performance: ACC 95.14% | EER 2.82% | F1 91.61%
🔧 Threshold: 0.526
```

### 4️⃣ best_rawnet (RawNet-based Model)
```
📊 Architecture: End-to-end raw waveform processing network
🎯 Specialty: Direct audio signal analysis without preprocessing
⚡ Performance: ACC 84.02% | EER 6.17%
🔧 Threshold: 0.562
```

### Model Performance Results

Our models were evaluated on a test dataset with the following results:

| Model | Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 Score | EER |
|-------|-----------|-----|-----|-----|-----|----------|-----------|--------|----------|------|
| **best_mel_cnn.pt** | 0.521 | 229 | 5 | 16 | 41 | **92.76%** | 89.15% | 71.79% | 84.86% | 4.10% |
| **best_lfcc** | 0.618 | 271 | 19 | 20 | 103 | 90.55% | 84.44% | 83.76% | 80.99% | 4.96% |
| **best_mel** | 0.526 | 419 | 11 | 13 | 51 | **95.14%** | 82.09% | 79.49% | **91.61%** | **2.82%** |
| **best_rawnet** | 0.562 | 290 | 38 | 35 | 94 | 84.02% | 71.18% | 72.88% | 72.22% | 6.17% |

#### 📊 Key Metrics Explained:
- **Accuracy (ACC)**: Overall correctness of predictions
- **Precision (PREC)**: Of all predicted deepfakes, how many were actually deepfakes
- **Recall (REC)**: Of all actual deepfakes, how many were detected
- **F1 Score**: Harmonic mean of precision and recall
- **EER (Equal Error Rate)**: Point where false acceptance rate equals false rejection rate (lower is better)

#### 🏆 Best Performers:
- **Highest Accuracy**: best_mel (95.14%)
- **Lowest EER**: best_mel (2.82%)
- **Best F1 Score**: best_mel (91.61%)
- **Primary Model**: best_mel_cnn.pt

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Browser Tab    │────▶│  Audio Capture  │────▶│  WAV Convert    │
│  (Any website)  │     │  (Tab Audio)    │     │  (16kHz Mono)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                            ┌─────────────────────┐
                                            │  🤗 Hugging Face    │
                                            │      API            │
                                            └────────┬────────────┘
                                                     │
                        ┌────────────────────────────┼────────────────────────────┐
                        │                            │                            │
                        ▼                            ▼                            ▼
              ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
              │   best_rawnet   │        │    best_lfcc    │        │    best_mel     │
              │   ACC: 84.02%   │        │   ACC: 90.55%   │        │   ACC: 95.14%   │
              └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
                       │                          │                          │
                       └──────────────────────────┼──────────────────────────┘
                                                  │
                                                  ▼
                                      ┌─────────────────────┐
                                      │  best_mel_cnn.pt ⭐ │
                                      │   (Primary Model)   │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │   Detection Result  │
                                      │   🤖 AI / 👤 Human  │
                                      └─────────────────────┘
```

---

## 🚀 How It Works

### Step 1: Audio Capture
```javascript
// Capture tab audio using Chrome's tabCapture API
stream = await navigator.mediaDevices.getUserMedia({
  audio: {
    mandatory: {
      chromeMediaSource: 'tab',
      chromeMediaSourceId: streamId
    }
  }
});
```

### Step 2: Audio Processing
```javascript
// Convert to 16kHz WAV for optimal model performance
const offCtx = new OfflineAudioContext(1, audio.duration * 16000, 16000);
```

### Step 3: API Prediction
```javascript
// Send to Hugging Face hosted model
const response = await fetch('url', {
  method: 'POST',
  headers: { 'x-api-key': API_KEY },
  body: formData
});
```

### Step 4: Display Result
```
🟢 OK  = Human voice detected
🔴 AI! = Deepfake/AI voice detected
🟡 ... = Analyzing
```

---

## 📊 Spectrogram Analysis

GLiTcH also provides visual analysis tools to compare audio:

### Real Human Speech vs AI-Generated

| Feature | Human 👤 | AI 🤖 |
|---------|----------|-------|
| **Pauses** | Natural breathing gaps | Continuous, no breaks |
| **Waveform** | Varied amplitude | Uniform patterns |
| **Spectrogram** | Irregular vertical bands | Dense, consistent energy |
| **Mel Spectrogram** | Organic variations | Repetitive horizontal bands |



---

## 🛠️ Installation

### Chrome Extension

1. Clone this repository
```bash
git clone https://github.com/jaindevshrut/GLiTcH.git
```

2. Open Chrome and go to `chrome://extensions`

3. Enable **Developer mode**

4. Click **Load unpacked** and select the `glitch-extension` folder

5. Pin the GLiTcH extension to your toolbar

### Python Analysis Tools

```bash
pip install librosa matplotlib numpy requests
```

---

## 📁 Project Structure

```
glitch-extension/
├── 📄 manifest.json        # Extension configuration
├── 📄 background.js        # Service worker (click handling)
├── 📄 offscreen.html       # Offscreen document for audio capture
├── 📄 offscreen.js         # Audio capture, WAV conversion, API calls
├── 📄 popup.html           # Extension popup UI
├── 📄 popup.css            # Popup styles
├── 📄 popup.js             # Popup functionality
├── 📁 icons/               # Extension icons
│   ├── icon16.svg
│   ├── icon48.svg
│   └── icon128.svg
└── 📁 huggingface-space/   # Hugging Face deployment
    ├── app.py              # Gradio/FastAPI backend
    ├── requirements.txt    # Python dependencies
    └── README.md           # Space documentation

```

---

## 🎮 Usage

### Basic Usage
1. Open any website with audio (YouTube, Google Meet, etc.)
2. Click the **GLiTcH** extension icon
3. Wait ~15 seconds for analysis
4. See the result badge:
   - 🟢 **OK** = Human
   - 🔴 **AI!** = Deepfake detected

### API Testing
```bash
python test.py
# Output: Result: spoofed (92.7% confidence)
```

---


## 🏆 Results & Performance

| Metric | Value |
|--------|-------|
| Detection Accuracy | ~93% |
| Processing Time | ~15 seconds |
| Supported Formats | WAV, MP3, WebM |
| Sample Rate | 16kHz (resampled) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **ASVspoof** - Anti-Spoofing datasets
- **Aurigin AI** - API hosting
- **Hugging Face** - Model deployment platform
- **Librosa** - Audio analysis library

---

<div align="center">

**Made with ❤️ to fight deepfakes**

🎭 GLiTcH - *Because truth matters*

</div>
