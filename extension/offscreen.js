let mediaStream = null;
let mediaRecorder = null;
let chunks = [];
let playbackAudioContext = null;
let playbackSourceNode = null;
let lastBlobUrl = null;
let recordingStartedAt = null;

const AURIGIN_API_KEY = "4Az2Tf1tz6525y1sFQtvw1QllU8qnkbh92sSWrYR";
const AURIGIN_PREDICT_URL = "https://api.aurigin.ai/v1/predict";
const ANALYSIS_TIMEOUT_MS = 20000;

function pickAudioMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm"];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function floatTo16BitPCM(outputView, offset, input) {
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    outputView.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function interleaveChannels(channelData) {
  const channelCount = channelData.length;
  const length = channelData[0]?.length ?? 0;
  const interleaved = new Float32Array(length * channelCount);
  let idx = 0;
  for (let i = 0; i < length; i++) {
    for (let ch = 0; ch < channelCount; ch++) {
      interleaved[idx++] = channelData[ch][i];
    }
  }
  return interleaved;
}

function audioBufferToWavBlob(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const channelData = Array.from({ length: numChannels }, (_, ch) =>
    audioBuffer.getChannelData(ch)
  );
  const interleaved = interleaveChannels(channelData);

  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = interleaved.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  floatTo16BitPCM(view, 44, interleaved);

  return new Blob([buffer], { type: "audio/wav" });
}

async function webmBlobToWavBlob(webmBlob) {
  const ctx = new AudioContext();
  const arrayBuffer = await webmBlob.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  try {
    await ctx.close();
  } catch {}
  return audioBufferToWavBlob(audioBuffer);
}

function timeoutAfter(ms) {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(`Timed out after ${ms}ms`)), ms);
  });
}

async function postToAurigin({ blob, filename, contentType, timeoutMs }) {
  const form = new FormData();
  form.append("file", new Blob([blob], { type: contentType }), filename);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const startedAt = Date.now();
  try {
    const resp = await fetch(AURIGIN_PREDICT_URL, {
      method: "POST",
      headers: { "x-api-key": AURIGIN_API_KEY },
      body: form,
      signal: controller.signal
    });

    const text = await resp.text().catch(() => "");
    if (!resp.ok) {
      return {
        ok: false,
        status: resp.status,
        error: text || `HTTP ${resp.status}`,
        debug: { ms: Date.now() - startedAt, status: resp.status, body: text?.slice?.(0, 800) }
      };
    }

    const json = text ? JSON.parse(text) : null;
    return {
      ok: true,
      json,
      debug: { ms: Date.now() - startedAt, status: resp.status, body: text?.slice?.(0, 800) }
    };
  } catch (e) {
    const errName = e?.name ?? "";
    const errMsg = e?.message ?? String(e);
    if (errName === "AbortError") {
      return { ok: false, status: 0, error: `Timed out after ${timeoutMs}ms`, debug: { ms: timeoutMs } };
    }
    return { ok: false, status: 0, error: errMsg, debug: { ms: Date.now() - startedAt } };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function analyzeRecording(recordingBlob, recordMs) {
  const analysisStartedAt = Date.now();
  const meta = {
    webmBytes: recordingBlob.size,
    webmType: recordingBlob.type || "audio/webm",
    timeoutMs: ANALYSIS_TIMEOUT_MS,
    recordMs: typeof recordMs === "number" ? recordMs : null
  };

  await chrome.runtime.sendMessage({
    type: "analysis-result",
    data: { analysis: { status: "pending", stage: "converting", debug: meta } }
  });

  const convertStartedAt = Date.now();
  const wavBlob = await Promise.race([
    webmBlobToWavBlob(recordingBlob),
    timeoutAfter(ANALYSIS_TIMEOUT_MS)
  ]);
  meta.wavBytes = wavBlob.size;
  meta.convertMs = Date.now() - convertStartedAt;

  await chrome.runtime.sendMessage({
    type: "analysis-result",
    data: { analysis: { status: "pending", stage: "uploading", debug: meta } }
  });

  const result = await postToAurigin({
    blob: wavBlob,
    filename: "recording.wav",
    contentType: "audio/wav",
    timeoutMs: ANALYSIS_TIMEOUT_MS
  });

  meta.http = result.debug || null;
  meta.totalMs = Date.now() - analysisStartedAt;
  if (meta.http?.ms && meta.wavBytes) {
    meta.uploadBytesPerSec = (meta.wavBytes * 1000) / meta.http.ms;
  }

  if (!result.ok) {
    const err = new Error(result.error || "Upload failed");
    err.debug = meta;
    throw err;
  }

  await chrome.runtime.sendMessage({
    type: "analysis-result",
    data: { analysis: { status: "pending", stage: "parsing", debug: meta } }
  });

  return { json: result.json, meta };
}

function pickFirst(obj, keys) {
  if (!obj || typeof obj !== "object") return undefined;
  for (const key of keys) {
    const val = obj[key];
    if (val !== undefined && val !== null) return val;
  }
  return undefined;
}

function normalizeLabel(value) {
  if (value === undefined || value === null) return null;
  const s = String(value).trim();
  if (!s) return null;
  return s;
}

function parseAuriginResult(json) {
  const root = json && typeof json === "object" ? json : null;
  const global = root?.global && typeof root.global === "object" ? root.global : null;
  const predictionId = normalizeLabel(root?.prediction_id);

  let label =
    normalizeLabel(root?.prediction) ||
    normalizeLabel(root?.verdict) ||
    normalizeLabel(root?.result) ||
    normalizeLabel(root?.label) ||
    normalizeLabel(pickFirst(global, ["prediction", "verdict", "result", "label", "decision", "class"]));

  const boolDeepfake = pickFirst(global, ["is_deepfake", "deepfake", "fake"]);
  if (!label && typeof boolDeepfake === "boolean") {
    label = boolDeepfake ? "deepfake" : "real";
  }

  let score = pickFirst(root, ["score", "confidence", "probability"]);
  if (typeof score !== "number") {
    score = pickFirst(global, ["score", "confidence", "probability", "fake_probability", "deepfake_probability"]);
  }

  if (typeof score !== "number" && global?.probabilities && typeof global.probabilities === "object") {
    const p = global.probabilities;
    const fake =
      typeof p.fake === "number"
        ? p.fake
        : typeof p.deepfake === "number"
          ? p.deepfake
          : typeof p.spoof === "number"
            ? p.spoof
            : null;
    if (typeof fake === "number") score = fake;
  }

  if (!label && typeof score === "number") {
    label = score >= 0.5 ? "deepfake" : "real";
  }

  if (!label) label = "unknown";

  return { label, score: typeof score === "number" ? score : null, predictionId };
}

async function startRecording(streamId) {
  if (mediaRecorder) return;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: streamId
      }
    },
    video: false
  });

  playbackAudioContext = new AudioContext();
  playbackSourceNode = playbackAudioContext.createMediaStreamSource(mediaStream);
  playbackSourceNode.connect(playbackAudioContext.destination);
  await playbackAudioContext.resume();

  chunks = [];
  const mimeType = pickAudioMimeType();
  mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  });

  mediaRecorder.addEventListener("stop", async () => {
    try {
      const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
      lastBlobUrl = URL.createObjectURL(blob);
      chrome.runtime.sendMessage({
        type: "recording-data",
        data: { blobUrl: lastBlobUrl, mimeType: blob.type }
      });

      (async () => {
        try {
          const recordMs =
            typeof recordingStartedAt === "number" ? Date.now() - recordingStartedAt : null;
          const { json, meta } = await analyzeRecording(blob, recordMs);
          const parsed = parseAuriginResult(json);
          const processingTime =
            typeof json?.processing_time === "number" ? json.processing_time : null;
          const audioDuration =
            typeof json?.audio_duration === "number" ? json.audio_duration : null;
          const rtf =
            typeof processingTime === "number" && typeof audioDuration === "number" && audioDuration > 0
              ? processingTime / audioDuration
              : null;
          const uploadMs = typeof meta?.http?.ms === "number" ? meta.http.ms : null;
          const uploadBytesPerSec =
            typeof meta?.uploadBytesPerSec === "number" ? meta.uploadBytesPerSec : null;
          const analysis = {
            status: "ok",
            label: parsed.label,
            score: parsed.score,
            raw: json,
            metrics: {
              recordMs,
              totalMs: typeof meta?.totalMs === "number" ? meta.totalMs : null,
              convertMs: typeof meta?.convertMs === "number" ? meta.convertMs : null,
              uploadMs,
              uploadBytesPerSec,
              wavBytes: typeof meta?.wavBytes === "number" ? meta.wavBytes : null,
              audioDurationSec: audioDuration,
              processingTimeSec: processingTime,
              rtf
            },
            debug: meta
          };
          analysis.debug = {
            ...analysis.debug,
            auriginUrl: AURIGIN_PREDICT_URL,
            responseKeys: json && typeof json === "object" ? Object.keys(json).slice(0, 50) : [],
            predictionId: parsed.predictionId || null
          };
          await chrome.runtime.sendMessage({
            type: "analysis-result",
            data: { analysis }
          });
        } catch (e) {
          chrome.runtime.sendMessage({
            type: "analysis-result",
            data: { analysis: { status: "error", error: e?.message ?? String(e), debug: e?.debug } }
          });
        }
      })();
    } catch (e) {
      chrome.runtime.sendMessage({
        type: "recording-error",
        data: { error: e?.message ?? String(e) }
      });
    } finally {
      cleanup();
    }
  });

  recordingStartedAt = Date.now();
  mediaRecorder.start(250);
}

function stopRecording() {
  if (!mediaRecorder) return;
  try {
    mediaRecorder.stop();
  } catch {
    cleanup();
  }
}

function cleanup() {
  if (playbackSourceNode) {
    try {
      playbackSourceNode.disconnect();
    } catch {}
  }

  playbackSourceNode = null;

  if (playbackAudioContext) {
    try {
      playbackAudioContext.close();
    } catch {}
  }

  playbackAudioContext = null;

  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      try {
        track.stop();
      } catch {}
    }
  }

  mediaStream = null;
  mediaRecorder = null;
  chunks = [];
}

chrome.runtime.onMessage.addListener((message) => {
  return (async () => {
    if (message?.target !== "offscreen") return;

    if (message.type === "offscreen-start") {
      await startRecording(message.data?.streamId);
      return { ok: true };
    }

    if (message.type === "offscreen-stop") {
      stopRecording();
      return { ok: true };
    }

    if (message.type === "offscreen-revoke-blob-url") {
      const blobUrl = message.data?.blobUrl;
      if (blobUrl) {
        try {
          URL.revokeObjectURL(blobUrl);
        } catch {}
      }
      if (blobUrl && blobUrl === lastBlobUrl) lastBlobUrl = null;
      return { ok: true };
    }
  })();
});
