"""
Voice Assistant - Main Orchestrator

State machine: IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE
- IDLE: Wake word detector is running, waiting for "hey jarvis"
- LISTENING: Recording user speech with VAD
- THINKING: Phi-4 processes audio, may execute tools
- SPEAKING: VibeVoice generates and plays speech response
"""
import sys
import time
import threading
import torch
from enum import Enum, auto

import config
from tool_registry import ToolRegistry
from tools.system_commands import register_system_tools
from tools.media_tools import register_media_tools
from tools.smart_home import register_smart_home_tools

from audio.wake_word import WakeWordDetector
from audio.audio_capture import AudioCapture
from ollama_brain import OllamaBrain
from audio.tts_engine import TTSEngine


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class VoiceAssistant:
    """Main orchestrator that ties all components together."""

    def __init__(self):
        self.state = State.IDLE
        self._wake_event = threading.Event()

        # Print GPU info
        print("=" * 60)
        print("VOICE ASSISTANT - Starting up...")
        print("=" * 60)
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print()

        # Initialize tool registry
        print("[Init] Setting up tool registry...")
        self.tool_registry = ToolRegistry()
        register_system_tools(self.tool_registry)
        register_media_tools(self.tool_registry)
        register_smart_home_tools(self.tool_registry)
        print(f"[Init] Registered {len(self.tool_registry.tools)} tools: {list(self.tool_registry.tools.keys())}")

        # Load models
        print("\n[Init] Loading Ollama Brain (STT + LLM)...")
        self.brain = OllamaBrain(self.tool_registry)

        print("\n[Init] Loading VibeVoice TTS...")
        self.tts = TTSEngine()

        # Initialize audio components (CPU-only, lightweight)
        print("\n[Init] Initializing audio capture (Silero VAD)...")
        self.audio_capture = AudioCapture()

        print("\n[Init] Initializing wake word detector...")
        self.wake_word = WakeWordDetector(on_wake_word=self._on_wake_word)

        # Print VRAM usage summary
        print("\n" + "=" * 60)
        print("VRAM USAGE:")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
                  f"{alloc:.2f} GB / {total:.1f} GB ({alloc/total*100:.0f}%)")
        print("=" * 60)
        print("\nReady! Say 'Hey Jarvis' to start.\n")

    def _on_wake_word(self):
        """Callback when wake word is detected."""
        if self.state == State.IDLE:
            self._wake_event.set()

    def run(self):
        """Main loop — run the assistant."""
        # Start wake word listener
        self.wake_word.start()

        try:
            while True:
                # ─── IDLE: Wait for wake word ───
                self.state = State.IDLE
                self._wake_event.clear()
                self._wake_event.wait()  # Blocks until wake word detected

                # ─── LISTENING: Record user speech ───
                self.state = State.LISTENING
                print("\n>>> Listening...")
                result = self.audio_capture.record()

                if result is None:
                    print(">>> No speech detected, going back to idle.")
                    continue

                audio_data, sample_rate = result

                # ─── THINKING: Process with Ollama ───
                self.state = State.THINKING
                print(">>> Thinking...")
                try:
                    response = self.brain.process_audio(audio_data, sample_rate)
                except Exception as e:
                    print(f"[Error] Brain processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    response = "Sorry, I had trouble processing that. Could you try again?"

                print(f">>> Response: {response}")

                # ─── SPEAKING: TTS output ───
                self.state = State.SPEAKING
                try:
                    self.tts.speak(response)
                except Exception as e:
                    print(f"[Error] TTS failed: {e}")
                    import traceback
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.wake_word.stop()
            print("Goodbye!")


def main():
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
