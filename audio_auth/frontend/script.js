
let mediaRecorder;
let audioChunks = [];
let capturedBlob = null;
let currentChallenge = "";

function openTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`button[onclick="openTab('${tabName}')"]`).classList.add('active');
}

async function startRecording(mode) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Try simple mime types if webm/opus not supported, but usually it is.
        // Chrome loves audio/webm
        mediaRecorder = new MediaRecorder(stream); 
        
        mediaRecorder.start();
        audioChunks = [];
        
        mediaRecorder.addEventListener("dataavailable", event => {
            audioChunks.push(event.data);
        });
        
        mediaRecorder.addEventListener("stop", () => {
            capturedBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(capturedBlob);
            const audioEl = document.getElementById(`${mode}-audio`);
            audioEl.src = audioUrl;
            audioEl.style.display = 'block';
            
            document.getElementById(`${mode}-submit`).disabled = false;
        });
        
        document.getElementById(`${mode}-start-btn`).disabled = true;
        document.getElementById(`${mode}-stop-btn`).disabled = false;
        document.getElementById(`${mode}-status`).innerText = "Recording... Speak clearly.";
        
    } catch (err) {
        console.error(err);
        alert("Error accessing microphone. Please allow microphone permissions.");
    }
}

function stopRecording(mode) {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        document.getElementById(`${mode}-start-btn`).disabled = false;
        document.getElementById(`${mode}-stop-btn`).disabled = true;
        document.getElementById(`${mode}-status`).innerText = "Recording stopped. Ready to submit.";
    }
}

async function registerUser() {
    const username = document.getElementById('reg-username').value;
    if (!username) return alert("Please enter a username");
    if (!capturedBlob) return alert("Please record audio");
    
    const formData = new FormData();
    formData.append("username", username);
    formData.append("audio", capturedBlob, "recording.webm");
    
    const statusEl = document.getElementById('reg-status');
    statusEl.innerText = "Registering... (This may take a moment to generate embeddings)";
    
    try {
        const res = await fetch("/api/register", {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        
        if (res.ok) {
            statusEl.innerText = "Success! " + data.message;
            statusEl.style.color = "lightgreen";
        } else {
            statusEl.innerText = "Error: " + data.detail;
            statusEl.style.color = "red";
        }
    } catch (err) {
        statusEl.innerText = "Network Error";
        console.error(err);
    }
}

async function getChallenge() {
    try {
        const res = await fetch("/api/challenge");
        const data = await res.json();
        currentChallenge = data.challenge;
        
        document.getElementById('challenge-text').innerText = `"${currentChallenge}"`;
        document.getElementById('challenge-box').classList.remove('hidden');
        document.getElementById('login-start-btn').disabled = false;
        
    } catch (err) {
        alert("Could not fetch challenge");
    }
}

async function loginUser() {
    const username = document.getElementById('login-username').value;
    if (!username) return alert("Please enter a username");
    if (!capturedBlob) return alert("Please record audio");
    if (!currentChallenge) return alert("Need a challenge first");
    
    const formData = new FormData();
    formData.append("username", username);
    formData.append("challenge", currentChallenge);
    formData.append("audio", capturedBlob, "recording.webm");
    
    const statusEl = document.getElementById('login-status');
    statusEl.innerText = "Verifying... (Analyzing voice and content)";
    statusEl.style.color = "white";
    
    try {
        const res = await fetch("/api/login", {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        
        if (res.ok) {
            statusEl.innerText = `Matched! ✅\nSpeaker Score: ${data.speaker_score.toFixed(2)}\nContent Score: ${data.content_score.toFixed(2)}`;
            statusEl.style.color = "lightgreen";
        } else {
            statusEl.innerText = `Login Failed ❌: ${data.detail}`;
            statusEl.style.color = "red";
        }
    } catch (err) {
        statusEl.innerText = "Network Error";
        console.error(err);
    }
}
