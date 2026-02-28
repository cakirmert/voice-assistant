#!/bin/bash
# WSL2 Setup Script for Phi-4 FP8 Inference Server
# Run this inside your WSL2 Ubuntu instance.
set -e

echo "=== Phi-4 FP8 Server Setup (WSL2) ==="

# 1. Create virtualenv
echo "[1/5] Creating Python venv..."
python3.12 -m venv venv || python3.10 -m venv venv || python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA
echo "[2/5] Installing PyTorch with CUDA 12.8..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install TensorRT-LLM
echo "[3/5] Installing TensorRT-LLM..."
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com

# 4. Install FastAPI + other deps
echo "[4/5] Installing FastAPI, uvicorn, and dependencies..."
pip install fastapi uvicorn soundfile numpy transformers accelerate peft

# 5. Download model (will cache in ~/.cache/huggingface)
echo "[5/5] Pre-downloading Phi-4 FP8 model..."
python -c "
from huggingface_hub import snapshot_download
print('Downloading nvidia/Phi-4-multimodal-instruct-FP8...')
snapshot_download('nvidia/Phi-4-multimodal-instruct-FP8')
print('Downloading microsoft/Phi-4-multimodal-instruct (processor only)...')
from transformers import AutoProcessor
AutoProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)
print('Done!')
"

echo ""
echo "=== Setup complete! ==="
echo "Start the server with:"
echo "  source venv/bin/activate"
echo "  uvicorn server:app --host 0.0.0.0 --port 8401"
echo ""
echo "Then on Windows, run: python main.py"
