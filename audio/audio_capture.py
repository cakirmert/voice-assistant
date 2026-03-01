"""
Audio Capture — Records speech after wake word, using Silero VAD to detect end of speech.
"""
import numpy as np
import pyaudio
import torch
import time
import sys
import os
# Add parent to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AudioCapture:
    """Records user speech with voice activity detection to know when they stop speaking."""

    def __init__(self):
        # Load Silero VAD
        print("[AudioCapture] Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = self.vad_utils
        print("[AudioCapture] Silero VAD loaded.")

    def record(self) -> tuple[np.ndarray, int] | None:
        """
        Record speech from the microphone until the user stops speaking.
        Returns (audio_numpy_int16, sample_rate) or None if nothing captured.
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=config.AUDIO_CHANNELS,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.AUDIO_CHUNK_SIZE,
        )

        print("[AudioCapture] Recording... speak now!")
        chunks = []
        silence_start = None
        speech_detected = False
        start_time = time.time()

        try:
            while True:
                elapsed = time.time() - start_time

                # Max duration guard
                if elapsed > config.VAD_MAX_DURATION:
                    print(f"[AudioCapture] Max duration ({config.VAD_MAX_DURATION}s) reached.")
                    break

                # Read audio chunk
                pcm = stream.read(config.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(pcm, dtype=np.int16)
                chunks.append(audio_chunk)

                # Convert to float32 for VAD
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float)

                # Run VAD
                try:
                    speech_prob = self.vad_model(audio_tensor, config.AUDIO_SAMPLE_RATE).item()
                except Exception:
                    speech_prob = 0.0

                if speech_prob > config.VAD_THRESHOLD:
                    speech_detected = True
                    silence_start = None
                else:
                    if speech_detected:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > config.VAD_SILENCE_DURATION:
                            print(f"[AudioCapture] Silence detected. Stopping. ({elapsed:.1f}s recorded)")
                            break

                # Timeout if no speech after 5 seconds
                if not speech_detected and elapsed > 5.0:
                    print("[AudioCapture] No speech detected. Cancelling.")
                    return None

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not chunks or not speech_detected:
            return None

        # Concatenate all chunks
        full_audio = np.concatenate(chunks)
        print(f"[AudioCapture] Captured {len(full_audio) / config.AUDIO_SAMPLE_RATE:.1f}s of audio")
        return full_audio, config.AUDIO_SAMPLE_RATE
