document.addEventListener('DOMContentLoaded', function() {
  // DOM Elements
  const container = document.getElementById('container');
  const statusRing = document.getElementById('status-ring');
  const statusIcon = document.getElementById('status-icon');
  const statusText = document.getElementById('status-text');
  const resultDisplay = document.getElementById('result-display');
  const resultLabel = document.getElementById('result-label');

  // Aurigin API Configuration
  const AURIGIN_API_URL = "https://api.aurigin.ai/v1/predict";
  const DEFAULT_API_KEY = "YOUR_API_KEY_HERE"; // Replace with your API key

  // State
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let timerInterval = null;
  let currentSeconds = 0;
  let audioStream = null;
  let lastAudioBlob = null;
  const CLIP_DURATION = 10;

  // Hide popup initially
  container.style.display = 'none';

  // Auto-start monitoring
  setTimeout(() => {
    startRecording();
  }, 500);

  // Start recording
  async function startRecording() {
    try {
      const isStreamActive = audioStream && 
                             audioStream.active && 
                             audioStream.getAudioTracks().some(track => track.readyState === 'live');

      if (!isStreamActive) {
        audioStream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000
          } 
        });
      }

      if (!audioStream) {
        console.error('Could not capture audio.');
        return;
      }

      isRecording = true;
      startRecordingCycle();

    } catch (error) {
      console.error('Audio capture error:', error);
    }
  }

  // Start a recording cycle
  function startRecordingCycle() {
    if (!isRecording || !audioStream) return;

    audioChunks = []; 
    currentSeconds = 0;
    
    const mimeType = getSupportedMimeType();
    
    try {
      mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: mimeType,
        audioBitsPerSecond: 128000
      });
    } catch (e) {
      mediaRecorder = new MediaRecorder(audioStream);
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      if (!isRecording) return;

      const audioBlob = new Blob(audioChunks, { type: mimeType });
      audioChunks = []; 
      
      const wavBlob = await convertToWav(audioBlob);
      lastAudioBlob = wavBlob;

      await sendToAuriginAPI(wavBlob);

      if (isRecording) {
        startRecordingCycle();
      }
    };

    mediaRecorder.start(1000);

    timerInterval = setInterval(() => {
      currentSeconds++;

      if (currentSeconds >= CLIP_DURATION) {
        clearInterval(timerInterval);
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }
    }, 1000);
  }

  // Convert audio blob to WAV format
  async function convertToWav(blob) {
    return new Promise((resolve) => {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const reader = new FileReader();
      
      reader.onload = async () => {
        try {
          const audioBuffer = await audioContext.decodeAudioData(reader.result);
          const wavBlob = audioBufferToWav(audioBuffer);
          resolve(wavBlob);
        } catch (e) {
          resolve(blob);
        }
      };
      
      reader.onerror = () => resolve(blob);
      reader.readAsArrayBuffer(blob);
    });
  }

  // Convert AudioBuffer to WAV Blob
  function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1;
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const samples = buffer.length;
    const dataSize = samples * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    
    const offset = 44;
    const channelData = [];
    for (let i = 0; i < numChannels; i++) {
      channelData.push(buffer.getChannelData(i));
    }
    
    for (let i = 0; i < samples; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset + (i * blockAlign) + (channel * bytesPerSample), intSample, true);
      }
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Get supported MIME type
  function getSupportedMimeType() {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4'
    ];
    
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return 'audio/webm';
  }

  // Send audio to Aurigin API
  async function sendToAuriginAPI(wavBlob) {
    const apiKey = DEFAULT_API_KEY;

    try {
      const formData = new FormData();
      formData.append('file', wavBlob, `recording_${Date.now()}.wav`);

      const response = await fetch(AURIGIN_API_URL, {
        method: 'POST',
        headers: {
          'x-api-key': apiKey
        },
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('API Response:', result);
      processAuriginResult(result);

    } catch (error) {
      console.error('Aurigin API Error:', error);
    }
  }

  // Process Aurigin API result
  function processAuriginResult(response) {
    let isAI = false;

    console.log('Processing API response:', response);

    if (typeof response === 'object') {
      // Check for Aurigin specific response format
      if (response.global && response.global.result) {
        isAI = response.global.result === 'spoofed';
        console.log('Aurigin result:', response.global.result, '-> isAI:', isAI);
      }
      else if (response.prediction !== undefined) {
        isAI = response.prediction === 1 || response.prediction === 'ai' || response.prediction === 'fake';
      } else if (response.result !== undefined) {
        isAI = response.result === 1 || response.result === 'ai' || response.result === 'fake' || response.result === 'spoofed';
      } else if (response.label !== undefined) {
        isAI = response.label === 1 || response.label === 'ai' || response.label === 'fake' || response.label === 'synthetic';
      } else if (response.is_ai !== undefined) {
        isAI = response.is_ai === true || response.is_ai === 1;
      } else if (response.fake !== undefined) {
        isAI = response.fake === true || response.fake === 1;
      } else if (response.score !== undefined) {
        isAI = response.score > 0.5;
      } else if (response.confidence !== undefined && response.class !== undefined) {
        isAI = response.class === 'ai' || response.class === 'fake' || response.class === 'synthetic';
      }
    } else if (typeof response === 'number') {
      isAI = response === 1 || response > 0.5;
    } else if (typeof response === 'string') {
      isAI = response === '1' || response.toLowerCase() === 'ai' || response.toLowerCase() === 'fake';
    }

    console.log('Final isAI determination:', isAI);

    // ONLY show popup if AI is detected
    if (isAI) {
      showAIDetected();
    } else {
      hidePopup();
    }
  }

  // Show AI detected popup
  function showAIDetected() {
    container.style.display = 'block';
    statusRing.classList.add('detecting');
    statusIcon.textContent = '🤖';
    statusText.textContent = 'AI Detected!';
    resultLabel.textContent = 'AI Voice Detected';
    resultDisplay.style.display = 'block';
  }

  // Hide popup
  function hidePopup() {
    container.style.display = 'none';
  }

  // Cleanup when popup closes
  window.addEventListener('beforeunload', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
    }
  });
});

  function updateSourceUI() {
    if (audioSource === 'mic') {
      sourceMicBtn.classList.add('active');
      sourceTabBtn.classList.remove('active');
      sourceHint.textContent = 'Capturing your microphone';
    } else {
      sourceMicBtn.classList.remove('active');
      sourceTabBtn.classList.add('active');
      sourceHint.textContent = 'Capturing audio from current tab (video call)';
    }
  }

  // Save API Key
  saveKeyBtn.addEventListener('click', function() {
    const key = apiKeyInput.value.trim();
    if (key) {
      chrome.storage.local.set({ auriginApiKey: key }, function() {
        saveKeyBtn.textContent = '✅';
        setTimeout(() => {
          saveKeyBtn.textContent = '💾';
        }, 1500);
      });
    }
  });

  // Play last clip
  playClipBtn.addEventListener('click', function() {
    if (lastAudioBlob) {
      if (audioContext && audioContext.state === 'running') {
        // If we are monitoring, use the existing context but be careful about feedback
        const audioUrl = URL.createObjectURL(lastAudioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      } else {
        // Use standard HTML5 audio
        const audioUrl = URL.createObjectURL(lastAudioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      }
    }
  });

  // Download last clip
  downloadClipBtn.addEventListener('click', function() {
    if (lastAudioBlob) {
      const url = URL.createObjectURL(lastAudioBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audio_clip_${Date.now()}.wav`;
      a.click();
      URL.revokeObjectURL(url);
    }
  });

  // Toggle recording
  toggleBtn.addEventListener('click', function() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // Start recording
  async function startRecording() {
    // Check API key only if not in demo mode
    if (!isDemoMode) {
      const apiKey = apiKeyInput.value.trim();
      if (!apiKey) {
        showError('Please enter the Aurigin API Key first, or enable Demo Mode');
        return;
      }
    }

    hideError();

    // DEMO MODE: No microphone needed - just simulate everything
    if (isDemoMode) {
      isRecording = true;
      updateUI();
      startDemoLoop();
      return;
    }

    // REAL MODE: Need actual audio capture
    try {
      // Check if we have an active stream we can reuse
      const isStreamActive = audioStream && 
                             audioStream.active && 
                             audioStream.getAudioTracks().some(track => track.readyState === 'live');

      if (!isStreamActive) {
        // We need a new stream
        if (audioSource === 'mic') {
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                } 
            });
        } else {
            // Capture tab audio (video call audio from the other person)
            // Tab capture always needs a new stream invocation
            audioStream = await captureTabAudio();
        }
      }

      if (!audioStream) {
        showError('Could not capture audio. Make sure you\'re on a video call page.');
        return;
      }

      // Playback audio if capturing tab (otherwise it's muted)
      if (audioSource === 'tab') {
          if (!audioContext) {
              audioContext = new (window.AudioContext || window.webkitAudioContext)();
          }
          if (audioContext.state === 'suspended') {
              await audioContext.resume();
          }
          // Create source only if not already created for this stream
          // Simplified: Just create a new connection chain.
          const source = audioContext.createMediaStreamSource(audioStream);
          source.connect(audioContext.destination);
      }

      isRecording = true;
      updateUI();

      // Start the recording loop
      startRecordingCycle();

    } catch (error) {
      console.error('Audio capture error:', error);
      if (error.name === 'NotAllowedError') {
        showError('Permission denied. Please allow audio access and try again.');
      } else {
        showError('Error capturing audio: ' + error.message);
      }
    }
  }

  // Capture tab audio (for video calls)
  async function captureTabAudio() {
    return new Promise((resolve, reject) => {
      chrome.tabCapture.capture({
        audio: true,
        video: false
      }, (stream) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (stream) {
          resolve(stream);
        } else {
          reject(new Error('Failed to capture tab audio'));
        }
      });
    });
  }

  // Start a recordings cycle
  function startRecordingCycle() {
    if (!isRecording || !audioStream) return;

    // CRITICAL: Clear previous chunks effectively to ensure fresh audio
    audioChunks = []; 
    currentSeconds = 0;
    updateTimer();
    
    // Log to console to verify new cycle
    console.log(`Starting new recording cycle at ${new Date().toISOString()}`);

    // Create MediaRecorder - try to use WAV-compatible format
    const mimeType = getSupportedMimeType();
    
    try {
      mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: mimeType,
        audioBitsPerSecond: 128000
      });
    } catch (e) {
      mediaRecorder = new MediaRecorder(audioStream);
    }

    mediaRecorder.ondataavailable = (event) => {
      // Only push valid data
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      if (!isRecording) return; // If stopped manually, ignore this implicit stop

      console.log(`Cycle finished with ${audioChunks.length} chunks`);

      // Create audio blob from CURRENT chunks
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      
      // Clear chunks immediately after creating blob to prevent any double-use
      const currentChunksSize = audioChunks.length;
      audioChunks = []; 
      
      // Convert to WAV for Aurigin API
      const wavBlob = await convertToWav(audioBlob);
      lastAudioBlob = wavBlob;

      // Show clip info
      const sizeKB = Math.round(wavBlob.size / 1024);
      clipSize.textContent = `Clip: ${sizeKB} KB (Fresh)`;
      clipInfo.classList.remove('hidden');

      // Store clip locally
      await storeClipLocally(wavBlob);

      // Process the clip
      if (isDemoMode) {
        simulateDemoResult();
      } else {
        await sendToAuriginAPI(wavBlob);
      }

      // Start next cycle if still recording and user hasn't pressed stop
      if (isRecording) {
        startRecordingCycle();
      }
    };

    // Start recording
    mediaRecorder.start(1000);

    // Timer update
    timerInterval = setInterval(() => {
      currentSeconds++;
      updateTimer();

      if (currentSeconds >= CLIP_DURATION) {
        clearInterval(timerInterval);
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }
    }, 1000);
  }

  // Convert audio blob to WAV format
  async function convertToWav(blob) {
    return new Promise((resolve) => {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const reader = new FileReader();
      
      reader.onload = async () => {
        try {
          const audioBuffer = await audioContext.decodeAudioData(reader.result);
          const wavBlob = audioBufferToWav(audioBuffer);
          resolve(wavBlob);
        } catch (e) {
          console.log('Could not convert to WAV, using original format');
          resolve(blob);
        }
      };
      
      reader.onerror = () => resolve(blob);
      reader.readAsArrayBuffer(blob);
    });
  }

  // Convert AudioBuffer to WAV Blob
  function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const samples = buffer.length;
    const dataSize = samples * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    
    // Write audio data
    const offset = 44;
    const channelData = [];
    for (let i = 0; i < numChannels; i++) {
      channelData.push(buffer.getChannelData(i));
    }
    
    for (let i = 0; i < samples; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset + (i * blockAlign) + (channel * bytesPerSample), intSample, true);
      }
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Store clip locally using IndexedDB
  async function storeClipLocally(blob) {
    return new Promise((resolve) => {
      const request = indexedDB.open('AudioClipsDB', 1);
      
      request.onerror = () => resolve();

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('clips')) {
          db.createObjectStore('clips', { keyPath: 'id', autoIncrement: true });
        }
      };

      request.onsuccess = (event) => {
        const db = event.target.result;
        const transaction = db.transaction(['clips'], 'readwrite');
        const store = transaction.objectStore('clips');
        
        store.add({
          timestamp: Date.now(),
          blob: blob,
          size: blob.size
        });

        const countRequest = store.count();
        countRequest.onsuccess = () => {
          if (countRequest.result > 10) {
            const cursorRequest = store.openCursor();
            cursorRequest.onsuccess = (e) => {
              const cursor = e.target.result;
              if (cursor) cursor.delete();
            };
          }
        };

        transaction.oncomplete = () => resolve();
      };
    });
  }

  // Get supported MIME type
  function getSupportedMimeType() {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4'
    ];
    
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return 'audio/webm';
  }

  // Stop recording
  function stopRecording() {
    isRecording = false;

    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }

    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    
    // Explicitly clear chunks
    audioChunks = [];
    
    // Close audio context to stop playback and free resources
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    // Don't stop the tracks here to keep permission 'hot'
    // if (audioStream) {
    //   audioStream.getTracks().forEach(track => track.stop());
    //   audioStream = null;
    // }

    mediaRecorder = null;
    updateUI();
  }

  // Cleanup on unload
  window.addEventListener('unload', () => {
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
  });

  // Demo Loop - simulates recording without actual microphone
  function startDemoLoop() {
    if (!isRecording) return;

    currentSeconds = 0;
    updateTimer();

    // Simulate recording progress
    timerInterval = setInterval(() => {
      if (!isRecording) {
        clearInterval(timerInterval);
        return;
      }

      currentSeconds++;
      updateTimer();

      if (currentSeconds >= CLIP_DURATION) {
        clearInterval(timerInterval);
        
        // Show fake clip size
        const fakeSize = Math.floor(50 + Math.random() * 100);
        clipSize.textContent = `Clip: ${fakeSize} KB (simulated)`;
        clipInfo.classList.remove('hidden');

        // Generate random result
        simulateDemoResult();

        // Start next cycle
        if (isRecording) {
          setTimeout(() => {
            if (isRecording) startDemoLoop();
          }, 2000);
        }
      }
    }, 1000);
  }

  // Demo mode: simulate random result
  function simulateDemoResult() {
    statusText.textContent = 'Analyzing audio...';

    setTimeout(() => {
      const result = Math.random() < 0.5 ? 0 : 1;
      
      totalClips++;
      clipsCount.textContent = totalClips;

      const isAI = result === 1;
      
      if (isAI) {
        totalAiDetections++;
        aiCount.textContent = totalAiDetections;
      }

      showResult(isAI);
      addHistoryItem(isAI, true);
      hideError();
    }, 500 + Math.random() * 1000);
  }

  // Send audio to Aurigin API
  async function sendToAuriginAPI(wavBlob) {
    const apiKey = apiKeyInput.value.trim() || DEFAULT_API_KEY;
    
    statusText.textContent = 'Sending to Aurigin AI...';

    try {
      // Create FormData with the WAV file
      const formData = new FormData();
      formData.append('file', wavBlob, `recording_${Date.now()}.wav`);

      statusText.textContent = 'Analyzing with AI model...';

      const response = await fetch(AURIGIN_API_URL, {
        method: 'POST',
        headers: {
          'x-api-key': apiKey
        },
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('Aurigin API Response:', result);
      
      processAuriginResult(result);

    } catch (error) {
      console.error('Aurigin API Error:', error);
      showError('API Error: ' + error.message);
      
      if (isRecording) {
        statusText.textContent = 'Recording...';
      }
    }
  }

  // Process Aurigin API result
  function processAuriginResult(response) {
    let isAI = false;

    // Handle different possible response formats from Aurigin
    // Adjust this based on actual API response structure
    if (typeof response === 'object') {
      
      // Check for nested global object (Specific to your API response)
      if (response.global && response.global.result) {
         isAI = response.global.result === 'spoofed';
         console.log('Aurigin Global Result:', response.global.result, 'isAI:', isAI);
      }
      // Common response fields to check
      else if (response.prediction !== undefined) {
        isAI = response.prediction === 1 || response.prediction === 'ai' || response.prediction === 'fake';
      } else if (response.result !== undefined) {
        isAI = response.result === 1 || response.result === 'ai' || response.result === 'fake' || response.result === 'spoofed';
      } else if (response.label !== undefined) {
        isAI = response.label === 1 || response.label === 'ai' || response.label === 'fake' || response.label === 'synthetic';
      } else if (response.is_ai !== undefined) {
        isAI = response.is_ai === true || response.is_ai === 1;
      } else if (response.fake !== undefined) {
        isAI = response.fake === true || response.fake === 1;
      } else if (response.score !== undefined) {
        // If score > 0.5, consider it AI
        isAI = response.score > 0.5;
      } else if (response.confidence !== undefined && response.class !== undefined) {
        isAI = response.class === 'ai' || response.class === 'fake' || response.class === 'synthetic';
      }
      
      // Log for debugging
      console.log('Parsed result - isAI:', isAI, 'from response:', response);
    } else if (typeof response === 'number') {
      isAI = response === 1 || response > 0.5;
    } else if (typeof response === 'string') {
      isAI = response === '1' || response.toLowerCase() === 'ai' || response.toLowerCase() === 'fake';
    }

    totalClips++;
    clipsCount.textContent = totalClips;

    if (isAI) {
      totalAiDetections++;
      aiCount.textContent = totalAiDetections;
    }

    showResult(isAI);
    addHistoryItem(isAI, false);
    hideError();
  }

  // Show result
  function showResult(isAI) {
    resultDisplay.classList.remove('hidden', 'ai-detected', 'ai-not-detected');
    statusRing.classList.remove('ai-detected', 'ai-not-detected', 'recording');

    if (isAI) {
      resultDisplay.classList.add('ai-detected');
      statusRing.classList.add('ai-detected');
      resultLabel.textContent = '🤖 AI Detected';
      statusIcon.textContent = '⚠️';
      statusText.textContent = 'AI Voice Detected!';
    } else {
      resultDisplay.classList.add('ai-not-detected');
      statusRing.classList.add('ai-not-detected');
      resultLabel.textContent = '✅ Human Voice';
      statusIcon.textContent = '✓';
      statusText.textContent = 'Human Voice Confirmed';
    }

    setTimeout(() => {
      if (isRecording) {
        statusRing.classList.remove('ai-detected', 'ai-not-detected');
        statusRing.classList.add('recording');
        statusIcon.textContent = '🎤';
        statusText.textContent = 'Recording...';
      }
    }, 2000);
  }

  // Add to history
  function addHistoryItem(isAI, isDemo) {
    historyContainer.classList.remove('hidden');

    const item = document.createElement('div');
    item.className = `history-item ${isAI ? 'ai-detected' : 'ai-not-detected'}`;
    
    const time = new Date().toLocaleTimeString();
    const modeLabel = isDemo ? ' (Demo)' : ' (Aurigin)';
    item.innerHTML = `
      <span class="dot"></span>
      <span class="time">${time}</span>
      <span class="label">${isAI ? '🤖 AI' : '✅ Human'}${modeLabel}</span>
    `;

    historyLog.insertBefore(item, historyLog.firstChild);

    while (historyLog.children.length > 20) {
      historyLog.removeChild(historyLog.lastChild);
    }
  }

  // Update timer display
  function updateTimer() {
    const percentage = (currentSeconds / CLIP_DURATION) * 100;
    timerProgress.style.width = `${percentage}%`;
    timerText.textContent = `${currentSeconds}s / ${CLIP_DURATION}s`;
  }

  // Update UI based on recording state
  function updateUI() {
    if (isRecording) {
      toggleBtn.classList.add('recording');
      btnIcon.textContent = '⏹️';
      btnText.textContent = 'Stop Monitoring';
      
      statusRing.classList.remove('idle');
      statusRing.classList.add('recording');
      statusIcon.textContent = '🎤';
      statusText.textContent = 'Recording...';
      
      timerContainer.classList.remove('hidden');
      stats.classList.remove('hidden');
      
      sourceMicBtn.disabled = true;
      sourceTabBtn.disabled = true;
      demoModeCheckbox.disabled = true;
      
    } else {
      toggleBtn.classList.remove('recording');
      btnIcon.textContent = '▶️';
      btnText.textContent = 'Start Monitoring';
      
      statusRing.classList.remove('recording', 'ai-detected', 'ai-not-detected');
      statusRing.classList.add('idle');
      statusIcon.textContent = '🎤';
      statusText.textContent = 'Ready to monitor';
      
      timerContainer.classList.add('hidden');
      timerProgress.style.width = '0%';
      currentSeconds = 0;
      
      sourceMicBtn.disabled = false;
      sourceTabBtn.disabled = false;
      demoModeCheckbox.disabled = false;
    }
  }

  // Show error
  function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
  }

  // Hide error
  function hideError() {
    errorDiv.classList.add('hidden');
  }
});
