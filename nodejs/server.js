const fs = require("fs");
const path = require("path");
const express = require("express");
const multer = require("multer");
const OpenAI = require("openai");

require("dotenv").config();

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 30 * 1024 * 1024 },
});

const config = {
  appTitle: process.env.APP_TITLE ?? "Voice Webchat",
  whisperUrl: process.env.WHISPER_SERVER_URL ?? "http://127.0.0.1:9191/inference",
  whisperTranslate:
    process.env.WHISPER_TRANSLATE === "true" || process.env.WHISPER_TRANSLATE === "1",
  llmBaseUrl: process.env.LLM_BASE_URL ?? "http://127.0.0.1:9090/v1",
  llmApiKey: process.env.LLM_API_KEY ?? "LAN",
  llmModel: process.env.LLM_MODEL ?? "qwen3",
  llmSystem: process.env.LLM_SYSTEM ?? "You are a helpful assistant.",
  llmPreprompt:
    process.env.LLM_PREPROMPT ??
    "Use the following transcription to respond conversationally.",
  llmPostprompt: process.env.LLM_POSTPROMPT ?? "",
  qwenTtsUrl: process.env.QWEN_TTS_URL ?? "http://127.0.0.1:8000",
  qwenTtsLang: process.env.QWEN_TTS_LANG ?? "Auto",
};

const referenceAudioPath = process.env.REF_AUDIO_PATH
  ? path.resolve(__dirname, process.env.REF_AUDIO_PATH)
  : path.resolve(__dirname, "..", "poppy.wav");
const referenceTextPath = process.env.REF_TEXT_PATH
  ? path.resolve(__dirname, process.env.REF_TEXT_PATH)
  : path.resolve(__dirname, "..", "poppy.txt");
const referenceAudioBuffer = fs.readFileSync(referenceAudioPath);
const referenceText = fs.readFileSync(referenceTextPath, "utf-8").trim();
if (!referenceText) {
  throw new Error("Reference text file is empty.");
}

const openai = new OpenAI({
  apiKey: config.llmApiKey,
  baseURL: config.llmBaseUrl,
});

app.get("/api/config", (req, res) => {
  res.json({ appTitle: config.appTitle });
});

app.use(express.static(path.join(__dirname, "public")));

async function transcribeAudio(audioBuffer, filename, mimeType, translate) {
  const form = new FormData();
  form.append(
    "file",
    new Blob([audioBuffer], { type: mimeType || "audio/webm" }),
    filename || "recording.webm"
  );
  const responseFormat = "text";
  form.append("response_format", responseFormat);
  if (translate) {
    form.append("translate", "true");
  }

  const response = await fetch(config.whisperUrl, {
    method: "POST",
    body: form,
  });

  const raw = await response.text();
  if (!response.ok) {
    throw new Error(`Whisper server error: ${response.status} ${raw}`);
  }

  if (responseFormat === "text") {
    return raw.trim();
  }

  try {
    const json = JSON.parse(raw);
    if (typeof json.text === "string") {
      return json.text.trim();
    }
  } catch (err) {
    // Ignore JSON parse errors and return raw text below.
  }

  return raw.trim();
}

function normalizeChatHistory(raw) {
  if (!raw) {
    return [];
  }
  let parsed = raw;
  if (typeof raw === "string") {
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      return [];
    }
  }
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed
    .filter((entry) => entry && typeof entry === "object")
    .map((entry) => {
      const role = entry.role === "assistant" ? "assistant" : entry.role === "user" ? "user" : null;
      const content = typeof entry.content === "string" ? entry.content.trim() : "";
      return role && content ? { role, content } : null;
    })
    .filter(Boolean);
}

async function generateAssistantReply(transcript, chatHistory) {
  const messages = [
    { role: "system", content: config.llmSystem },
    { role: "user", content: config.llmPreprompt },
  ];

  if (chatHistory.length > 0) {
    messages.push(...chatHistory);
  }
  messages.push({ role: "user", content: transcript });

  if (config.llmPostprompt) {
    messages.push({ role: "user", content: config.llmPostprompt });
  }

  const completion = await openai.chat.completions.create({
    model: config.llmModel,
    messages,
    temperature: 0.7,
  });

  const reply = completion.choices?.[0]?.message?.content?.trim();
  if (!reply) {
    throw new Error("LLM response was empty.");
  }
  return reply;
}

async function synthesizeSpeech(text, refAudioBuffer, refText, synLang) {
  const payload = {
    ref_audio_base64: refAudioBuffer.toString("base64"),
    ref_text: refText,
    syn_text: text,
    syn_lang: synLang,
  };

  const response = await fetch(`${config.qwenTtsUrl.replace(/\/$/, "")}/clone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const responseBody = await response.text();
  if (!response.ok) {
    throw new Error(`TTS server error: ${response.status} ${responseBody}`);
  }

  const json = JSON.parse(responseBody);
  if (!json.audio_base64) {
    throw new Error("TTS response missing audio_base64.");
  }
  return json.audio_base64;
}

app.post("/api/chat", upload.single("audio"), async (req, res) => {
  try {
    const audioFile = req.file;
    const synLang = (config.qwenTtsLang || "Auto").trim() || "Auto";
    const translate = config.whisperTranslate;
    const chatHistory = normalizeChatHistory(req.body.chatHistory);

    if (!audioFile) {
      return res.status(400).json({ error: "Missing recorded audio." });
    }

    const transcript = await transcribeAudio(
      audioFile.buffer,
      audioFile.originalname,
      audioFile.mimetype,
      translate
    );
    const assistantReply = await generateAssistantReply(transcript, chatHistory);
    const audioBase64 = await synthesizeSpeech(
      assistantReply,
      referenceAudioBuffer,
      referenceText,
      synLang
    );

    return res.json({
      transcript,
      assistantReply,
      audioBase64,
    });
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error.",
    });
  }
});

const port = process.env.PORT ? Number(process.env.PORT) : 3000;
app.listen(port, "0.0.0.0", () => {
  console.log(`Webchat server listening on http://0.0.0.0:${port}`);
});
