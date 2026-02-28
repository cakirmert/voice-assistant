"""
Phi-4 Brain - HTTP client that sends audio to the WSL2 inference server.
Falls back to local Transformers model if server is unreachable.
"""
import io
import json
import base64
import numpy as np
import soundfile as sf
import requests
from tool_registry import ToolRegistry
import config


# Server URL (WSL2 localhost)
PHI4_SERVER_URL = getattr(config, "PHI4_SERVER_URL", "http://localhost:8401")


class Phi4Brain:
    """
    Sends audio to the Phi-4 inference server running in WSL2.
    Falls back to loading locally if the server is not available.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.server_url = PHI4_SERVER_URL
        self.local_model = None
        self.local_processor = None
        self.local_gen_config = None
        self._mode = None  # "server" or "local"
        self._connect()

    def _connect(self):
        """Check if the WSL2 server is running."""
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"[Phi4Brain] Connected to WSL2 server")
                print(f"[Phi4Brain]   Backend: {info.get('backend', '?')}")
                print(f"[Phi4Brain]   GPU: {info.get('gpu', '?')}")
                print(f"[Phi4Brain]   VRAM: {info.get('vram_gb', '?')} GB")
                self._mode = "server"
                return
        except Exception as e:
            print(f"[Phi4Brain] WSL2 server not available: {e}")

        print("[Phi4Brain] Falling back to local model loading...")
        self._load_local_model()

    def _load_local_model(self):
        """Load Phi-4 locally as fallback."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        print(f"[Phi4Brain] Loading local model: {config.PHI4_MODEL_ID}")
        self.local_processor = AutoProcessor.from_pretrained(
            config.PHI4_MODEL_ID, trust_remote_code=True,
        )

        attn_impl = config.PHI4_ATTN_IMPL
        try:
            self.local_model = AutoModelForCausalLM.from_pretrained(
                config.PHI4_MODEL_ID,
                device_map=config.PRIMARY_GPU,
                torch_dtype=config.PHI4_TORCH_DTYPE,
                trust_remote_code=True,
                _attn_implementation=attn_impl,
            )
        except Exception as e:
            print(f"[Phi4Brain] flash_attention_2 failed ({e}), using eager")
            attn_impl = "eager"
            self.local_model = AutoModelForCausalLM.from_pretrained(
                config.PHI4_MODEL_ID,
                device_map=config.PRIMARY_GPU,
                torch_dtype=config.PHI4_TORCH_DTYPE,
                trust_remote_code=True,
                _attn_implementation=attn_impl,
            )

        self.local_model.eval()
        self.local_gen_config = GenerationConfig.from_pretrained(config.PHI4_MODEL_ID)
        self._mode = "local"

        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[Phi4Brain] Local model loaded. VRAM: {allocated:.2f} GB (attn: {attn_impl})")

    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Process audio — route to server or local model."""
        if self._mode == "server":
            return self._server_process_audio(audio_data, sample_rate)
        else:
            return self._local_process_audio(audio_data, sample_rate)

    def _server_process_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Send audio to WSL2 server over HTTP."""
        # Convert to float and encode as WAV bytes
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        else:
            audio_float = audio_data.astype(np.float64)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sample_rate, format="WAV")
        wav_bytes = wav_buffer.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        # Build request
        payload = {
            "audio_b64": audio_b64,
            "system_prompt": config.SYSTEM_PROMPT,
            "tools_json": self.tool_registry.get_tool_definitions_json(),
            "max_new_tokens": config.PHI4_MAX_NEW_TOKENS,
        }

        try:
            r = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=120,
            )
            result = r.json()

            if result.get("error"):
                print(f"[Phi4Brain] Server error: {result['error']}")
                return "Sorry, I had trouble processing that."

            response = result.get("text", "")
            print(f"[Phi4Brain] Server response: {response[:200]}")

            # Check for tool calls
            tool_calls = self.tool_registry.parse_tool_calls(response)
            if tool_calls:
                return self._handle_tool_calls(tool_calls)

            return response

        except requests.Timeout:
            print("[Phi4Brain] Server request timed out")
            return "Sorry, the server took too long to respond."
        except requests.ConnectionError:
            print("[Phi4Brain] Server connection lost, switching to local")
            self._load_local_model()
            return self._local_process_audio(audio_data, sample_rate)

    def _local_process_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Process audio locally with Transformers."""
        import torch

        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float64) / 32768.0
        else:
            audio_float = audio_data.astype(np.float64)

        tools_block = ""
        tools_json = self.tool_registry.get_tool_definitions_json()
        if tools_json != "[]":
            tools_block = "<|tool|>" + tools_json + "<|/tool|>"

        prompt = (
            "<|system|>\n" + config.SYSTEM_PROMPT + "\n"
            + tools_block + "\n<|end|>\n"
            + "<|user|>\n<|audio_1|> Respond to what the user said.\n"
            + "<|end|>\n"
            + "<|assistant|>\n"
        )

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sample_rate, format="WAV")
        wav_buffer.seek(0)
        audio_for_proc, sr = sf.read(wav_buffer)

        inputs = self.local_processor(
            text=prompt, audios=[(audio_for_proc, sr)], return_tensors="pt",
        ).to(config.PRIMARY_GPU)

        with torch.no_grad():
            output_ids = self.local_model.generate(
                **inputs,
                max_new_tokens=config.PHI4_MAX_NEW_TOKENS,
                generation_config=self.local_gen_config,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.local_processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        print(f"[Phi4Brain] Local response: {response[:200]}")

        tool_calls = self.tool_registry.parse_tool_calls(response)
        if tool_calls:
            return self._handle_tool_calls(tool_calls)

        return response

    def _handle_tool_calls(self, tool_calls: list) -> str:
        """Execute tool calls and generate a natural summary."""
        results = []
        for call in tool_calls:
            tool_name = call.get("name", "unknown")
            print(f"[Phi4Brain] Executing tool: {tool_name}")
            result = self.tool_registry.execute_tool_call(json.dumps(call))
            results.append(f"Tool '{tool_name}' returned: {result}")
            print(f"[Phi4Brain] Tool result: {result[:200]}")

        tool_results_str = "\n".join(results)
        return self.process_text(
            f"I used tools to help the user. Results:\n"
            f"{tool_results_str}\n"
            f"Please provide a brief, conversational summary."
        )

    def process_text(self, text: str) -> str:
        """Process text-only query."""
        if self._mode == "server":
            return self._server_process_text(text)
        else:
            return self._local_process_text(text)

    def _server_process_text(self, text: str) -> str:
        """Send text query to server."""
        payload = {
            "text": text,
            "system_prompt": config.SYSTEM_PROMPT,
            "tools_json": self.tool_registry.get_tool_definitions_json(),
            "max_new_tokens": config.PHI4_MAX_NEW_TOKENS,
        }
        try:
            r = requests.post(f"{self.server_url}/infer", json=payload, timeout=120)
            result = r.json()
            return result.get("text", "")
        except Exception as e:
            print(f"[Phi4Brain] Server text request failed: {e}")
            return "Sorry, I couldn't process that."

    def _local_process_text(self, text: str) -> str:
        """Generate from text locally."""
        import torch

        prompt = (
            "<|system|>\n" + config.SYSTEM_PROMPT + "\n<|end|>\n"
            + "<|user|>\n" + text + "\n<|end|>\n"
            + "<|assistant|>\n"
        )
        inputs = self.local_processor(text=prompt, return_tensors="pt").to(config.PRIMARY_GPU)

        with torch.no_grad():
            output_ids = self.local_model.generate(
                **inputs,
                max_new_tokens=config.PHI4_MAX_NEW_TOKENS,
                generation_config=self.local_gen_config,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        return self.local_processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
