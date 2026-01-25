const recordButton = document.getElementById("recordButton");
const statusEl = document.getElementById("status");
const chatLog = document.getElementById("chatLog");
const clearChatButton = document.getElementById("clearChatButton");
const securityNote = document.getElementById("securityNote");
const appTitleEl = document.getElementById("appTitle");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
const chatHistory = [];

const preferredMimeTypes = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

async function loadAppConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    if (data && typeof data.appTitle === "string" && data.appTitle.trim()) {
      const title = data.appTitle.trim();
      document.title = title;
      if (appTitleEl) {
        appTitleEl.textContent = title;
      }
    }
  } catch (error) {
    console.error(error);
  }
}

function getSupportedMimeType() {
  if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) {
    return "";
  }
  return preferredMimeTypes.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function extensionForMimeType(mimeType) {
  if (!mimeType) {
    return "webm";
  }
  if (mimeType.includes("ogg")) {
    return "ogg";
  }
  if (mimeType.includes("wav")) {
    return "wav";
  }
  if (mimeType.includes("webm")) {
    return "webm";
  }
  return "webm";
}

function setStatus(message) {
  statusEl.textContent = message;
}

function updateRecordButton() {
  recordButton.textContent = isRecording ? "Stop chatting" : "Start chatting";
}

function scrollChatToBottom() {
  if (!chatLog) {
    return;
  }
  requestAnimationFrame(() => {
    chatLog.scrollTop = chatLog.scrollHeight;
  });
}

function renderChat() {
  if (!chatLog) {
    return;
  }
  chatLog.innerHTML = "";
  if (!chatHistory.length) {
    const empty = document.createElement("p");
    empty.className = "chat-empty";
    empty.textContent = "No messages yet.";
    chatLog.appendChild(empty);
    scrollChatToBottom();
    return;
  }
  chatHistory.forEach((entry) => {
    const messageEl = document.createElement("div");
    messageEl.className = `chat-message ${entry.role}`;
    const roleEl = document.createElement("div");
    roleEl.className = "chat-message-role";
    roleEl.textContent = entry.role === "user" ? "You" : "Assistant";
    const contentEl = document.createElement("div");
    contentEl.className = "chat-message-content";
    contentEl.textContent = entry.content;

    messageEl.append(roleEl, contentEl);

    if (entry.role === "assistant" && entry.audioBase64) {
      const audioEl = document.createElement("audio");
      audioEl.controls = true;
      audioEl.src = `data:audio/wav;base64,${entry.audioBase64}`;
      messageEl.appendChild(audioEl);
    }

    chatLog.appendChild(messageEl);
  });
  scrollChatToBottom();
}

function appendChatMessage(role, content, audioBase64 = null) {
  if (!content) {
    return;
  }
  chatHistory.push({ role, content, audioBase64 });
  renderChat();
}

function clearChat() {
  chatHistory.length = 0;
  renderChat();
}

function playLatestAssistantAudio() {
  if (!chatLog) {
    return Promise.resolve();
  }
  const audioElements = chatLog.querySelectorAll("audio");
  const latestAudio = audioElements[audioElements.length - 1];
  if (!latestAudio) {
    return Promise.resolve();
  }
  return latestAudio.play().catch(() => {});
}

function isSecureOrigin() {
  return window.isSecureContext;
}

function formatMicError(error) {
  if (error && typeof error === "object" && "name" in error) {
    const name = error.name;
    if (name === "NotAllowedError") {
      return "Microphone access was blocked. Allow it in the browser permissions.";
    }
    if (name === "NotFoundError") {
      return "No microphone was found. Check your audio input.";
    }
    if (name === "NotReadableError") {
      return "The microphone is in use by another application.";
    }
    if (name === "SecurityError") {
      return "Microphone access requires HTTPS or localhost.";
    }
  }
  return "Failed to access microphone.";
}

if (!isSecureOrigin()) {
  if (securityNote) {
    securityNote.hidden = false;
  }
  recordButton.disabled = true;
  setStatus("Microphone access requires HTTPS or localhost.");
}

async function startRecording() {
  if (isRecording) {
    return;
  }
  let stream;
  setStatus("Requesting microphone access...");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    throw new Error(formatMicError(error));
  }
  const preferredMimeType = getSupportedMimeType();
  const options = preferredMimeType ? { mimeType: preferredMimeType } : undefined;
  mediaRecorder = new MediaRecorder(stream, options);

  audioChunks = [];
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      audioChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((track) => track.stop());
  };

  mediaRecorder.start();
  isRecording = true;
  updateRecordButton();
  setStatus("Listening...");
}

async function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }

  const stopPromise = new Promise((resolve) => {
    const handleStop = () => {
      mediaRecorder.removeEventListener("stop", handleStop);
      resolve();
    };
    mediaRecorder.addEventListener("stop", handleStop);
  });

  setStatus("Finalizing recording...");
  mediaRecorder.stop();
  await stopPromise;

  isRecording = false;
  updateRecordButton();

  const mimeType = mediaRecorder.mimeType || "audio/webm";
  const extension = extensionForMimeType(mimeType);
  const audioBlob = new Blob(audioChunks, { type: mimeType });
  await sendRecording(audioBlob, `recording.${extension}`);
}

async function sendRecording(audioBlob, filename) {
  const resolvedFilename = filename || "recording.webm";
  const formData = new FormData();
  formData.append("audio", audioBlob, resolvedFilename);
  const chatHistoryPayload = chatHistory.map(({ role, content }) => ({ role, content }));
  formData.append("chatHistory", JSON.stringify(chatHistoryPayload));

  try {
    setStatus("Uploading audio...");
    const transcribeResponsePromise = fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });

    setStatus("Transcribing audio...");
    const transcribeResponse = await transcribeResponsePromise;
    const transcribeData = await transcribeResponse.json();
    if (!transcribeResponse.ok) {
      throw new Error(transcribeData.error || "Request failed.");
    }

    const transcript =
      typeof transcribeData.transcript === "string" ? transcribeData.transcript : "";

    if (!transcript) {
      throw new Error("Transcription was empty.");
    }

    appendChatMessage("user", transcript);

    setStatus("Thinking about it...");
    const llmResponse = await fetch("/api/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, chatHistory: chatHistoryPayload }),
    });
    const llmData = await llmResponse.json();
    if (!llmResponse.ok) {
      throw new Error(llmData.error || "Request failed.");
    }

    const assistantReply =
      typeof llmData.assistantReply === "string" ? llmData.assistantReply : "";

    if (!assistantReply) {
      throw new Error("LLM response was empty.");
    }

    setStatus("Finding the words...");
    const ttsResponse = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: assistantReply }),
    });
    const ttsData = await ttsResponse.json();
    if (!ttsResponse.ok) {
      throw new Error(ttsData.error || "Request failed.");
    }

    const audioBase64 =
      typeof ttsData.audioBase64 === "string" ? ttsData.audioBase64 : null;

    appendChatMessage("assistant", assistantReply, audioBase64);
    if (audioBase64) {
      await playLatestAssistantAudio();
    }

    setStatus("Ready.");
  } catch (error) {
    console.error(error);
    setStatus(error instanceof Error ? error.message : "Request failed.");
  }
}

recordButton.addEventListener("click", () => {
  const action = isRecording ? stopRecording : startRecording;
  action().catch((error) => {
    console.error(error);
    setStatus(
      error instanceof Error ? error.message : "Failed to start or stop recording."
    );
  });
});

if (clearChatButton) {
  clearChatButton.addEventListener("click", () => {
    clearChat();
  });
}

loadAppConfig();
updateRecordButton();
renderChat();
