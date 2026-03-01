import sys
from tool_registry import ToolRegistry
from tools.system_commands import register_system_tools
from ollama_brain import OllamaBrain

def test_text_inference():
    print("--- Initializing Test Environment ---")
    
    # 1. Setup Registry
    registry = ToolRegistry()
    register_system_tools(registry)
    print(f"Registered {len(registry.tools)} tools: {list(registry.tools.keys())}")
    
    # 2. Setup Brain
    print("\n--- Initializing Ollama Brain ---")
    brain = OllamaBrain(registry)
    
    # 3. Get prompt
    # Use command line argument if provided, otherwise default to "open spotify"
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
    else:
        test_prompt = "open spotify please"
        
    print(f"\n--- Testing Prompt ---")
    print(f"USER: '{test_prompt}'")
    print("\n--- Processing ---")
    
    # 4. Run through process_text directly (bypasses audio/STT)
    final_response = brain.process_text(test_prompt)
    
    print("\n--- Final Output ---")
    print(f"ASSISTANT: {final_response}")

if __name__ == "__main__":
    test_text_inference()
