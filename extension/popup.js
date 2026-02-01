const toggleButton = document.getElementById("toggle");
const statusEl = document.getElementById("status");
const apiUrlInput = document.getElementById("apiUrl");
const resultEl = document.getElementById("result");
const metricsEl = document.getElementById("metrics");
const debugEl = document.getElementById("debug");

function setStatus(text) {
  statusEl.textContent = text ?? "";
}

function setResult(text) {
  resultEl.textContent = text ?? "";
}

function setMetrics(text) {
  metricsEl.textContent = text ?? "";
}

function setDebug(text) {
  const t = text ?? "";
  debugEl.hidden = !t;
  debugEl.textContent = t;
}

function fmtMs(ms) {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return null;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtBytesPerSec(bps) {
  if (typeof bps !== "number" || !Number.isFinite(bps) || bps <= 0) return null;
  const kbps = bps / 1024;
  if (kbps < 1024) return `${kbps.toFixed(1)} KB/s`;
  return `${(kbps / 1024).toFixed(2)} MB/s`;
}

function setUi({ isRecording, busy }) {
  toggleButton.disabled = Boolean(busy);
  toggleButton.textContent = isRecording ? "Stop recording" : "Start recording";
  setStatus(isRecording ? "Recording…" : "");
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] ?? null;
}

async function getStatus() {
  const res = await chrome.runtime.sendMessage({ type: "get-status" });
  if (!res || typeof res !== "object") return { isRecording: false, recordingTabId: null };
  return {
    isRecording: Boolean(res.isRecording),
    recordingTabId: res.recordingTabId ?? null
  };
}

async function startRecording() {
  const tab = await getActiveTab();
  if (!tab?.id) throw new Error("No active tab.");

  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id
  });

  await chrome.runtime.sendMessage({
    type: "start-recording",
    data: { streamId, tabId: tab.id }
  });
}

async function stopRecording() {
  await chrome.runtime.sendMessage({ type: "stop-recording" });
}

async function loadSettings() {
  const { apiUrl } = await chrome.storage.local.get({ apiUrl: "" });
  apiUrlInput.value = apiUrl || "";
}

async function saveSettings() {
  const apiUrl = String(apiUrlInput.value || "").trim();
  await chrome.storage.local.set({ apiUrl });
  await chrome.runtime.sendMessage({ type: "api-config-updated" });
}

async function loadLastAnalysis() {
  const res = await chrome.runtime.sendMessage({ type: "get-analysis" });
  if (!res || typeof res !== "object" || !res.analysis) {
    setResult("");
    setDebug("");
    setMetrics("");
    return;
  }
  const { analysis } = res;
  if (analysis.status === "error") {
    setResult(`Deepfake result: error (${analysis.error || "unknown"})`);
    setDebug(analysis.debug ? JSON.stringify(analysis.debug, null, 2) : "");
    setMetrics("");
    return;
  }
  if (analysis.status === "ok") {
    const label = analysis.label ?? "unknown";
    const score =
      typeof analysis.score === "number" ? ` (score: ${analysis.score})` : "";
    setResult(`Deepfake result: ${label}${score}`);
    setDebug(analysis.debug ? JSON.stringify(analysis.debug, null, 2) : "");

    const m = analysis.metrics && typeof analysis.metrics === "object" ? analysis.metrics : {};
    const parts = [];
    const total = fmtMs(m.totalMs)/3;
    const conv = fmtMs(m.convertMs);
    const up = fmtMs(m.uploadMs);
    const thr = fmtBytesPerSec(m.uploadBytesPerSec);
    const rtf = typeof m.rtf === "number" && Number.isFinite(m.rtf) ? m.rtf.toFixed(2) : null;
    const dur =
      typeof m.audioDurationSec === "number" && Number.isFinite(m.audioDurationSec)
        ? m.audioDurationSec.toFixed(2)
        : null;

    if (total) parts.push(`latency ${total}`);
    if (conv) parts.push(`convert ${conv}`);
    if (up) parts.push(`upload ${up}`);
    if (thr) parts.push(`throughput ${thr}`);
    if (rtf) parts.push(`rtf ${rtf}`);
    if (dur) parts.push(`audio ${dur}s`);

    setMetrics(parts.join(" | "));
    return;
  }
  if (analysis.status === "pending") {
    const stage = analysis.stage ? ` (${analysis.stage})` : "";
    setResult(`Deepfake result: analyzing…${stage}`);
    setDebug(analysis.debug ? JSON.stringify(analysis.debug, null, 2) : "");

    const d = analysis.debug && typeof analysis.debug === "object" ? analysis.debug : {};
    const parts = [];
    const conv = fmtMs(d.convertMs);
    const up = fmtMs(d.http?.ms);
    const thr = fmtBytesPerSec(d.uploadBytesPerSec);
    if (conv) parts.push(`convert ${conv}`);
    if (up) parts.push(`upload ${up}`);
    if (thr) parts.push(`throughput ${thr}`);
    setMetrics(parts.join(" | "));
    return;
  }
  setResult("");
  setDebug(analysis.debug ? JSON.stringify(analysis.debug, null, 2) : "");
  setMetrics("");
}

async function refresh() {
  const { isRecording } = await getStatus();
  setUi({ isRecording, busy: false });
  await loadLastAnalysis();
}

toggleButton.addEventListener("click", async () => {
  try {
    const { isRecording } = await getStatus();
    setUi({ isRecording, busy: true });
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
    await refresh();
  } catch (e) {
    setUi({ isRecording: false, busy: false });
    setStatus(e?.message ?? String(e));
  }
});

apiUrlInput.addEventListener("change", () => {
  saveSettings().catch(() => {});
});

Promise.all([loadSettings(), refresh()]).catch(() => {});

let pollTimer = null;
function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(() => {
    loadLastAnalysis().catch(() => {});
  }, 500);
}

function stopPolling() {
  if (!pollTimer) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

startPolling();
window.addEventListener("unload", stopPolling);
