"""
Combined pipeline test — loads LLM + TTS on separate GPUs, pipes output.
Phi-4 on cuda:0 (5060 Ti), Qwen3-TTS on cuda:1 (1660 Super).
"""
import sys
import time
import torch

sys.path.insert(0, ".")
import config
from tool_registry import ToolRegistry
from tools.system_commands import register_system_tools

print("=" * 60)
print("PIPELINE TEST: LLM (cuda:0) + TTS (cuda:1)")
print("=" * 60)

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"  GPU {i}: {name} ({total:.1f} GB)")

# Step 1: Load TTS first (smaller model, faster)
print("\n[1/3] Loading Qwen3-TTS on cuda:1...")
t0 = time.time()
from tts_engine import TTSEngine
tts = TTSEngine()
t1 = time.time()
print(f"  TTS loaded in {t1-t0:.1f}s")
print(f"  cuda:1 VRAM: {torch.cuda.memory_allocated(1) / (1024**3):.2f} GB")

# Step 2: Load LLM (local fallback, base model BF16)
print("\n[2/3] Loading Phi-4 on cuda:0...")
t0 = time.time()
registry = ToolRegistry()
register_system_tools(registry)
from phi4_brain import Phi4Brain
brain = Phi4Brain(registry)
t1 = time.time()
print(f"  LLM loaded in {t1-t0:.1f}s")
print(f"  cuda:0 VRAM: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")

# Summary
print("\n--- VRAM Summary ---")
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / (1024**3)
    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"  GPU {i}: {alloc:.2f} GB / {total:.1f} GB ({alloc/total*100:.0f}%)")

# Step 3: Pipeline test — LLM generates text, TTS speaks it
print("\n[3/3] Pipeline test: LLM → TTS")
print("  Sending text query to Phi-4...")
t0 = time.time()
response = brain.process_text("Say hello and introduce yourself in one short sentence.")
t1 = time.time()
print(f"  LLM response ({t1-t0:.1f}s): {response}")

print("  Sending to TTS...")
t0 = time.time()
tts.speak(response)
t1 = time.time()
print(f"  TTS done ({t1-t0:.1f}s)")

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE!")
print("=" * 60)
