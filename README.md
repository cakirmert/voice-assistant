# AI Voice Assistant

An always-listening, locally-running AI voice assistant using:
- **Phi-4-multimodal-instruct** (STT + LLM) — speech-to-text and reasoning
- **Qwen3-TTS-1.7B** / **VibeVoice-0.5B** (TTS) — configurable text-to-speech
- **openWakeWord** — "Hey Jarvis" wake word detection

## Architecture

```
Windows (native audio)                    WSL2 (TensorRT-LLM)
┌──────────────────────┐         HTTP    ┌─────────────────────┐
│ Mic → openWakeWord   │                 │ FastAPI server      │
│ → Silero VAD         │  base64 WAV     │ Phi-4 FP8 (TRT-LLM)│
│ → phi4_brain (client)├────────────────►│ /infer endpoint     │
│ ← response text      │◄────────────────┤ → text response     │
│ → TTS (configurable) │                 └─────────────────────┘
│ → Speaker            │     Falls back to local Transformers
└──────────────────────┘     if server unreachable
```

## Hardware Requirements

| Component | GPU | VRAM |
|-----------|-----|------|
| Phi-4 FP8 (TRT-LLM via WSL2) | Primary GPU | ~8.7 GB |
| Phi-4 BF16 (local fallback) | Primary GPU | ~12 GB |
| Qwen3-TTS 1.7B (FP16) | Secondary GPU | ~3.4 GB |
| VibeVoice 0.5B (BF16) | Secondary GPU | ~2 GB |
| openWakeWord | CPU | ~50 MB RAM |

## Quick Start

### 1. Windows Setup

```bash
# Create venv with Python 3.10
py -3.10 -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install transformers accelerate peft soundfile scipy pillow numpy
pip install pyaudio openwakeword psutil requests
pip install -U qwen-tts

# Clone and install VibeVoice (optional, for vibevoice backend)
git clone https://github.com/microsoft/VibeVoice.git
pip install -e VibeVoice
```

### 2. WSL2 Server (for TRT-LLM acceleration)

```bash
cd /mnt/d/AI/voice-assistant/wsl2_server
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8401
```

### 3. Run

```powershell
cd d:\AI\voice-assistant
.\venv\Scripts\activate
python main.py
```

Say **"Hey Jarvis"** to activate!

## Configuration

Edit `config.py` to change:
- **TTS backend**: `TTS_BACKEND = "qwen3"` or `"vibevoice"`
- **Wake word threshold**: `WAKE_WORD_THRESHOLD = 0.5`
- **Speaker voice**: `QWEN3_TTS_SPEAKER = "Ryan"`
- **Server URL**: `PHI4_SERVER_URL = "http://localhost:8401"`

## Tools

The assistant can execute tools when asked. Built-in tools:
- `run_command` — execute shell commands
- `get_current_time` — get date/time
- `get_system_info` — CPU, RAM, disk usage
- `open_application` — launch apps (notepad, calculator, browser, etc.)

Add custom tools in `tools/` and register them in `tools/system_commands.py`.

## License

MIT
