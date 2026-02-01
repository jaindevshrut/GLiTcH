const OFFSCREEN_URL = "offscreen.html";

let isRecording = false;
let recordingTabId = null;
let creatingOffscreen = null;
let lastAnalysis = null;

async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_URL);
  if ("getContexts" in chrome.runtime) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [offscreenUrl]
    });
    if (contexts.length > 0) return;
  } else {
    const has = await chrome.offscreen.hasDocument();
    if (has) return;
  }

  if (creatingOffscreen) {
    await creatingOffscreen;
    return;
  }

  creatingOffscreen = chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ["USER_MEDIA"],
    justification: "Record tab audio using MediaRecorder"
  });
  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

function sanitizeFilenamePart(part) {
  return String(part).replace(/[<>:"/\\|?*\u0000-\u001F]/g, "-");
}

async function downloadRecording({ arrayBuffer, mimeType }) {
  const blob = new Blob([arrayBuffer], { type: mimeType || "audio/webm" });
  const url = URL.createObjectURL(blob);
  await downloadRecordingFromUrl({ url, mimeType });
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

async function downloadRecordingFromUrl({ url, mimeType }) {
  const iso = new Date().toISOString();
  const ext = String(mimeType || "").includes("webm") ? "webm" : "webm";
  const filename = `recording-${sanitizeFilenamePart(iso)}.${ext}`;

  await chrome.downloads.download({
    url,
    filename,
    saveAs: false
  });
}

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.local.get({ apiUrl: "" });
  return String(apiUrl || "").trim();
}

async function setLastAnalysis(analysis) {
  lastAnalysis = analysis ?? null;
  await chrome.storage.local.set({ lastAnalysis });
}

async function loadLastAnalysis() {
  const { lastAnalysis: stored } = await chrome.storage.local.get({ lastAnalysis: null });
  lastAnalysis = stored ?? null;
}

loadLastAnalysis().catch(() => {});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    try {
      if (!message?.type) {
        sendResponse(undefined);
        return;
      }

      if (message.type === "get-status") {
        sendResponse({ isRecording, recordingTabId });
        return;
      }

      if (message.type === "get-analysis") {
        sendResponse({ analysis: lastAnalysis });
        return;
      }

      if (message.type === "get-api-config") {
        const apiUrl = await getApiUrl();
        sendResponse({ apiUrl });
        return;
      }

      if (message.type === "start-recording") {
        if (isRecording) {
          sendResponse({ isRecording, recordingTabId });
          return;
        }

        const { streamId, tabId } = message.data || {};
        if (!streamId) throw new Error("Missing streamId.");

        await ensureOffscreenDocument();
        isRecording = true;
        recordingTabId = tabId ?? null;
        await setLastAnalysis({ status: "idle" });
        chrome.runtime.sendMessage({
          type: "offscreen-start",
          target: "offscreen",
          data: { streamId }
        });
        sendResponse({ isRecording, recordingTabId });
        return;
      }

      if (message.type === "stop-recording") {
        if (!isRecording) {
          sendResponse({ isRecording, recordingTabId });
          return;
        }

        isRecording = false;
        await setLastAnalysis({ status: "pending" });
        chrome.runtime.sendMessage({
          type: "offscreen-stop",
          target: "offscreen"
        });
        sendResponse({ isRecording, recordingTabId });
        return;
      }

      if (message.type === "analysis-result") {
        const analysis = message.data?.analysis ?? null;
        await setLastAnalysis(analysis);
        sendResponse({ ok: true });
        return;
      }

      if (message.type === "recording-data") {
        const { blobUrl, arrayBuffer, mimeType } = message.data || {};
        if (blobUrl) {
          await downloadRecordingFromUrl({ url: blobUrl, mimeType });
          chrome.runtime.sendMessage({
            type: "offscreen-revoke-blob-url",
            target: "offscreen",
            data: { blobUrl }
          });
        } else if (arrayBuffer) {
          await downloadRecording({ arrayBuffer, mimeType });
        }
        isRecording = false;
        recordingTabId = null;
        sendResponse({ ok: true });
        return;
      }

      if (message.type === "recording-error") {
        isRecording = false;
        recordingTabId = null;
        await setLastAnalysis({ status: "error", error: message.data?.error || "Unknown error" });
        sendResponse({ ok: false, error: message.data?.error || "Unknown error" });
        return;
      }

      if (message.type === "api-config-updated") {
        sendResponse({ ok: true });
        return;
      }

      sendResponse(undefined);
    } catch (e) {
      isRecording = false;
      recordingTabId = null;
      await setLastAnalysis({ status: "error", error: e?.message ?? String(e) });
      sendResponse({ ok: false, error: e?.message ?? String(e) });
    }
  })();

  return true;
});
