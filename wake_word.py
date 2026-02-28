"""
Wake Word Detection — Always-on listening using openWakeWord.
Runs on CPU, triggers audio capture when "hey jarvis" is detected.
"""
import numpy as np
import pyaudio
import threading
import time
from openwakeword.model import Model as OWWModel
import config


class WakeWordDetector:
    """Continuously listens for a wake word using openWakeWord on CPU."""

    def __init__(self, on_wake_word: callable):
        """
        Args:
            on_wake_word: Callback fired when wake word is detected.
        """
        self.on_wake_word = on_wake_word
        self.running = False
        self._thread = None
        self._cooldown = 2.0  # seconds between detections

        # Initialize openWakeWord
        print("[WakeWord] Loading openWakeWord model...")
        self.model = OWWModel(
            wakeword_models=[config.WAKE_WORD],
            inference_framework="onnx",
        )
        print(f"[WakeWord] Loaded. Listening for '{config.WAKE_WORD}'")

    def start(self):
        """Start listening in a background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _listen_loop(self):
        """Main loop: capture mic audio and check for wake word."""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=config.AUDIO_CHANNELS,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.AUDIO_CHUNK_SIZE,
        )

        last_trigger = 0
        print("[WakeWord] Mic stream open. Listening...")

        try:
            while self.running:
                # Read audio chunk
                pcm = stream.read(config.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(pcm, dtype=np.int16)

                # Feed to openWakeWord
                self.model.predict(audio_data)

                # Check predictions
                for model_name, score in self.model.prediction_buffer.items():
                    current_score = score[-1] if len(score) > 0 else 0
                    if current_score > config.WAKE_WORD_THRESHOLD:
                        now = time.time()
                        if now - last_trigger > self._cooldown:
                            last_trigger = now
                            print(f"[WakeWord] Detected! (score={current_score:.2f})")
                            self.model.reset()
                            self.on_wake_word()
                            break
        except Exception as e:
            print(f"[WakeWord] Error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
