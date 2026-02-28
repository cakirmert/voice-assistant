"""
Phi-4 Inference Server - Runs inside WSL2 with TensorRT-LLM (or Transformers fallback).
Receives audio as WAV bytes over HTTP, returns text response.

Run:  uvicorn server:app --host 0.0.0.0 --port 8401
"""
import io
import json
import base64
import logging
from typing import Annotated

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Body
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phi4-server")

# --- Configuration ---
MODEL_ID_TRT = "nvidia/Phi-4-multimodal-instruct-FP8"
MODEL_ID_HF = "microsoft/Phi-4-multimodal-instruct"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 512

# --- Phi-4 tokens ---
TAG_SYS_S = "<|system|>"
TAG_END = "<|end|>"
TAG_USR_S = "<|user|>"
TAG_ASST_S = "<|assistant|>"
TAG_TOOL_S = "<|tool|>"
TAG_TOOL_E = "<|/tool|>"
TAG_AUDIO = "<|audio_1|>"

# --- Global model state ---
model = None
processor = None
gen_config = None
backend = None  # "trtllm" or "transformers"


class InferenceRequest(BaseModel):
    """Request body for inference."""
    audio_b64: str | None = None
    text: str | None = None
    system_prompt: str = ""
    tools_json: str = "[]"
    max_new_tokens: int = MAX_NEW_TOKENS


class InferenceResponse(BaseModel):
    """Response body from inference."""
    text: str
    error: str | None = None


app = FastAPI(title="Phi-4 Inference Server", version="1.0.0")


@app.on_event("startup")
def load_model():
    """Load Phi-4 model on startup. Tries TRT-LLM first, falls back to Transformers."""
    global model, processor, gen_config, backend

    # Try TensorRT-LLM first
    try:
        from tensorrt_llm import LLM
        logger.info(f"Loading TensorRT-LLM model: {MODEL_ID_TRT}")
        model = LLM(model=MODEL_ID_TRT, trust_remote_code=True)
        backend = "trtllm"
        logger.info("TensorRT-LLM backend loaded")
    except (ImportError, Exception) as e:
        logger.warning(f"TensorRT-LLM not available ({e}), falling back to Transformers")
        backend = "transformers"

    if backend == "transformers":
        from transformers import AutoModelForCausalLM, GenerationConfig
        logger.info(f"Loading Transformers model: {MODEL_ID_HF}")
        attn = "flash_attention_2"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID_HF,
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                _attn_implementation=attn,
            )
        except Exception:
            attn = "eager"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID_HF,
                device_map=DEVICE,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                _attn_implementation=attn,
            )
        model.eval()
        gen_config = GenerationConfig.from_pretrained(MODEL_ID_HF)
        logger.info(f"Transformers backend loaded (attn: {attn})")

    # Always load the processor for audio handling
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID_HF, trust_remote_code=True)

    alloc = torch.cuda.memory_allocated(0) / (1024**3)
    logger.info(f"VRAM used: {alloc:.2f} GB | Backend: {backend}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": backend,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
    }


@app.post("/infer")
def infer(request: InferenceRequest) -> InferenceResponse:
    """Run inference on audio or text input."""
    try:
        # Build prompt
        tools_block = TAG_TOOL_S + request.tools_json + TAG_TOOL_E if request.tools_json != "[]" else ""
        audio_input = None

        if request.audio_b64:
            # Decode audio from base64 WAV
            wav_bytes = base64.b64decode(request.audio_b64)
            wav_buffer = io.BytesIO(wav_bytes)
            audio_np, sr = sf.read(wav_buffer)
            audio_input = (audio_np, sr)

            prompt = (
                TAG_SYS_S + "\n" + request.system_prompt + "\n"
                + tools_block + "\n" + TAG_END + "\n"
                + TAG_USR_S + "\n" + TAG_AUDIO + " Respond to what the user said.\n"
                + TAG_END + "\n"
                + TAG_ASST_S + "\n"
            )
        elif request.text:
            prompt = (
                TAG_SYS_S + "\n" + request.system_prompt + "\n"
                + tools_block + "\n" + TAG_END + "\n"
                + TAG_USR_S + "\n" + request.text + "\n"
                + TAG_END + "\n"
                + TAG_ASST_S + "\n"
            )
        else:
            return InferenceResponse(text="", error="No audio or text provided")

        # Inference
        if backend == "trtllm":
            response_text = _infer_trtllm(prompt, audio_input, request.max_new_tokens)
        else:
            response_text = _infer_transformers(prompt, audio_input, request.max_new_tokens)

        return InferenceResponse(text=response_text)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return InferenceResponse(text="", error=str(e))


def _infer_trtllm(prompt: str, audio_input: tuple | None, max_tokens: int) -> str:
    """Inference using TensorRT-LLM."""
    from tensorrt_llm import SamplingParams

    if audio_input:
        inputs = processor(
            text=prompt,
            audios=[audio_input],
            return_tensors="pt",
        )
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    # TRT-LLM generate
    outputs = model.generate(inputs["input_ids"], sampling_params=sampling)
    response = processor.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True)
    return response.strip()


def _infer_transformers(prompt: str, audio_input: tuple | None, max_tokens: int) -> str:
    """Inference using HuggingFace Transformers."""
    if audio_input:
        inputs = processor(
            text=prompt,
            audios=[audio_input],
            return_tensors="pt",
        ).to(DEVICE)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            generation_config=gen_config,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return response.strip()
