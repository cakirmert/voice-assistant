"""
Test script for Phi-4 LLM text inference.
Tests the brain in local fallback mode (no WSL2 server needed).
"""
import sys
sys.path.insert(0, ".")
from tool_registry import ToolRegistry
from tools.system_commands import register_system_tools
from phi4_brain import Phi4Brain

print("=" * 60)
print("TEST: Phi-4 LLM Text Inference")
print("=" * 60)

# Setup tool registry
registry = ToolRegistry()
register_system_tools(registry)
print(f"Registered tools: {list(registry.tools.keys())}")

# Initialize brain (will fall back to local since no WSL2 server)
brain = Phi4Brain(registry)

# Test 1: Simple question
print("\n--- Test 1: Simple question ---")
response = brain.process_text("What is 2 + 2? Answer in one sentence.")
print(f"Response: {response}")

# Test 2: Tool-eligible question
print("\n--- Test 2: Tool-eligible question ---")
response = brain.process_text("What time is it right now?")
print(f"Response: {response}")

print("\n" + "=" * 60)
print("LLM TEST COMPLETE")
print("=" * 60)
