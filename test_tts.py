"""
Test script for TTS (Qwen3-TTS or VibeVoice, based on config).
Tests speech generation and playback independently.
"""
import sys
sys.path.insert(0, ".")
import config
from tts_engine import TTSEngine

print("=" * 60)
print(f"TEST: TTS Engine ({config.TTS_BACKEND})")
print("=" * 60)

# Load TTS
tts = TTSEngine()

# Test 1: Short sentence
print("\n--- Test 1: Short sentence ---")
tts.speak("Hello! I am Jarvis, your AI assistant.")

# Test 2: Longer response
print("\n--- Test 2: Longer response ---")
tts.speak("The current CPU usage is at 15 percent, and you have 24 gigabytes of RAM available.")

print("\n" + "=" * 60)
print("TTS TEST COMPLETE")
print("=" * 60)
