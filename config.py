"""
Voice Assistant Configuration
"""
import torch

# ─── GPU Configuration ───────────────────────────────────────────────
PRIMARY_GPU = "cuda:0"      # RTX 5060 Ti 16GB — Phi-4 multimodal
SECONDARY_GPU = "cuda:1"    # GTX 1660 Super 6GB — VibeVoice TTS

# ─── LLM Brain (GGUF) ─────────────────────────────
LLM_MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLM_MAX_NEW_TOKENS = 256
LOCAL_SERVER_URL = "http://localhost:8401"  # Native Windows inference server

# ─── STT Engine (Faster Whisper) ─────────────────────────
STT_MODEL_SIZE = "small.en"

# ─── TTS Engine ─────────────────────────────────────────────────────
TTS_BACKEND = "qwen3"       # "qwen3" or "vibevoice"

# Qwen3-TTS settings
QWEN3_TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN3_TTS_TORCH_DTYPE = torch.float16  # 1660 Super only supports FP16 (no BF16)
QWEN3_TTS_SPEAKER = "Ryan"
QWEN3_TTS_LANGUAGE = "English"

# VibeVoice settings
VIBEVOICE_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
VIBEVOICE_TORCH_DTYPE = torch.bfloat16
VIBEVOICE_SPEAKER = "Wayne"
VIBEVOICE_CFG_SCALE = 1.5
VIBEVOICE_SAMPLE_RATE = 24000

# ─── Wake Word ───────────────────────────────────────────────────────
WAKE_WORD = "hey_jarvis"    # openWakeWord model name
WAKE_WORD_THRESHOLD = 0.5   # Confidence threshold (0-1)

# ─── Audio Settings ──────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000   # 16kHz for Phi-4 input
AUDIO_CHANNELS = 1          # Mono
AUDIO_CHUNK_SIZE = 1280     # ~80ms at 16kHz (openWakeWord requirement)
AUDIO_FORMAT_WIDTH = 2      # 16-bit (2 bytes per sample)

# ─── Voice Activity Detection ────────────────────────────────────────
VAD_THRESHOLD = 0.5         # Silero VAD threshold
VAD_SILENCE_DURATION = 1.5  # Seconds of silence before stopping recording
VAD_MAX_DURATION = 40       # Max recording duration (seconds) — Phi-4 limit

# ─── System Prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Jarvis, a helpful voice assistant. You respond concisely and naturally.
When the user asks you to perform an action (run a command, open an app, check system info, etc.),
use the available tools. Keep responses brief and conversational — they will be spoken aloud."""
