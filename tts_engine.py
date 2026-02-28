"""
TTS Engine — Configurable between Qwen3-TTS and VibeVoice.
Loads on the secondary GPU (GTX 1660 Super) and plays audio through speakers.
"""
import os
import copy
import glob
import numpy as np
import torch
import pyaudio
import soundfile as sf
import config


class TTSEngine:
    """
    Configurable TTS engine. Supports:
    - "qwen3": Qwen3-TTS-12Hz-1.7B-CustomVoice (~3.4GB FP16)
    - "vibevoice": VibeVoice-Realtime-0.5B (~2GB BF16)
    """

    def __init__(self):
        self.backend = config.TTS_BACKEND
        self._model = None
        self._processor = None
        self._voice_cache = None
        self._sample_rate = None

        print(f"[TTS] Backend: {self.backend}")
        if self.backend == "qwen3":
            self._load_qwen3()
        elif self.backend == "vibevoice":
            self._load_vibevoice()
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}. Use 'qwen3' or 'vibevoice'.")

    # ─── Qwen3-TTS ───────────────────────────────────────────────────

    def _load_qwen3(self):
        """Load Qwen3-TTS-CustomVoice on the secondary GPU."""
        from qwen_tts import Qwen3TTSModel

        print(f"[TTS] Loading {config.QWEN3_TTS_MODEL_ID}...")
        print(f"[TTS] Device: {config.SECONDARY_GPU}, dtype: {config.QWEN3_TTS_TORCH_DTYPE}")

        attn = "flash_attention_2"
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                config.QWEN3_TTS_MODEL_ID,
                device_map=config.SECONDARY_GPU,
                dtype=config.QWEN3_TTS_TORCH_DTYPE,
                attn_implementation=attn,
            )
        except Exception as e:
            print(f"[TTS] flash_attention_2 failed ({e}), using sdpa")
            attn = "sdpa"
            self._model = Qwen3TTSModel.from_pretrained(
                config.QWEN3_TTS_MODEL_ID,
                device_map=config.SECONDARY_GPU,
                dtype=config.QWEN3_TTS_TORCH_DTYPE,
                attn_implementation=attn,
            )

        gpu_idx = int(config.SECONDARY_GPU.split(":")[-1])
        allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
        print(f"[TTS] Qwen3-TTS loaded. VRAM: {allocated:.2f} GB (attn: {attn})")
        print(f"[TTS] Speaker: {config.QWEN3_TTS_SPEAKER}, Language: {config.QWEN3_TTS_LANGUAGE}")

    def _speak_qwen3(self, text: str):
        """Generate and play speech using Qwen3-TTS."""
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=config.QWEN3_TTS_LANGUAGE,
            speaker=config.QWEN3_TTS_SPEAKER,
        )
        if wavs and len(wavs) > 0:
            self._play_audio(wavs[0], sr)
            duration = len(wavs[0]) / sr
            print(f"[TTS] Finished speaking ({duration:.1f}s)")
        else:
            print("[TTS] No audio output generated")

    # ─── VibeVoice ───────────────────────────────────────────────────

    def _load_vibevoice(self):
        """Load VibeVoice on the secondary GPU."""
        from vibevoice import (
            VibeVoiceStreamingForConditionalGenerationInference,
            VibeVoiceStreamingProcessor,
        )

        print(f"[TTS] Loading {config.VIBEVOICE_MODEL_ID}...")
        self._processor = VibeVoiceStreamingProcessor.from_pretrained(
            config.VIBEVOICE_MODEL_ID
        )

        attn = "flash_attention_2"
        try:
            self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config.VIBEVOICE_MODEL_ID,
                torch_dtype=config.VIBEVOICE_TORCH_DTYPE,
                device_map=config.SECONDARY_GPU,
                attn_implementation=attn,
            )
        except Exception as e:
            print(f"[TTS] flash_attention_2 failed ({e}), using sdpa")
            attn = "sdpa"
            self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config.VIBEVOICE_MODEL_ID,
                torch_dtype=config.VIBEVOICE_TORCH_DTYPE,
                device_map=config.SECONDARY_GPU,
                attn_implementation=attn,
            )

        self._model.eval()
        self._model.set_ddpm_inference_steps(num_steps=5)
        self._sample_rate = config.VIBEVOICE_SAMPLE_RATE

        # Load voice preset
        self._load_vibevoice_preset()

        gpu_idx = int(config.SECONDARY_GPU.split(":")[-1])
        allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
        print(f"[TTS] VibeVoice loaded. VRAM: {allocated:.2f} GB (attn: {attn})")

    def _load_vibevoice_preset(self):
        """Load voice preset for VibeVoice."""
        voices_dir = os.path.join(
            os.path.dirname(__file__), "VibeVoice", "demo", "voices", "streaming_model"
        )
        if not os.path.exists(voices_dir):
            print(f"[TTS] Voice presets not found at {voices_dir}")
            self._voice_cache = None
            return

        pt_files = glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True)
        voice_map = {}
        for pt_file in pt_files:
            name = os.path.splitext(os.path.basename(pt_file))[0].lower()
            voice_map[name] = os.path.abspath(pt_file)

        speaker = config.VIBEVOICE_SPEAKER.lower()
        device = config.SECONDARY_GPU
        if speaker in voice_map:
            self._voice_cache = torch.load(voice_map[speaker], map_location=device, weights_only=False)
            print(f"[TTS] Voice preset loaded: {speaker}")
        elif voice_map:
            first = list(voice_map.keys())[0]
            self._voice_cache = torch.load(voice_map[first], map_location=device, weights_only=False)
            print(f"[TTS] Using fallback voice: {first}")
        else:
            self._voice_cache = None

    def _speak_vibevoice(self, text: str):
        """Generate and play speech using VibeVoice."""
        if self._voice_cache is not None:
            inputs = self._processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=copy.deepcopy(self._voice_cache),
                padding=True, return_tensors="pt", return_attention_mask=True,
            )
        else:
            inputs = self._processor(
                text=text, padding=True, return_tensors="pt", return_attention_mask=True,
            )

        device = config.SECONDARY_GPU
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=config.VIBEVOICE_CFG_SCALE,
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=(
                    copy.deepcopy(self._voice_cache) if self._voice_cache else None
                ),
            )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else np.array(audio)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            self._play_audio(audio_np, self._sample_rate)
            duration = len(audio_np) / self._sample_rate
            print(f"[TTS] Finished speaking ({duration:.1f}s)")
        else:
            print("[TTS] No audio output generated")

    # ─── Common ──────────────────────────────────────────────────────

    def speak(self, text: str):
        """Convert text to speech and play through speakers."""
        if not text or not text.strip():
            return

        text = text.strip()
        print(f"[TTS] Speaking ({self.backend}): {text[:100]}...")

        try:
            if self.backend == "qwen3":
                self._speak_qwen3(text)
            else:
                self._speak_vibevoice(text)
        except Exception as e:
            print(f"[TTS] Error: {e}")
            import traceback
            traceback.print_exc()

    def _play_audio(self, audio_np: np.ndarray, sample_rate: int):
        """Play numpy audio through speakers via PyAudio."""
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.nan_to_num(audio_np, nan=0.0)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
        else:
            audio_int16 = audio_np.astype(np.int16)

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        try:
            stream.write(audio_int16.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
