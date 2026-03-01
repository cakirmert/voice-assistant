import config
from tool_registry import ToolRegistry
from audio.stt_engine import STTEngine

logger = logging.getLogger("ollama_brain")

OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "gpt-oss:20b"

class OllamaBrain:
    """
    Handles audio ingestion (via WSL STT) and inference via Ollama (gpt-oss-20b).
    Executes OpenAI-compatible tool calls.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.stt_engine = STTEngine()
        
        # Check if Ollama is running
        try:
            r = requests.get("http://localhost:11434/")
            if r.status_code == 200:
                print(f"[OllamaBrain] Connected to Ollama backend.")
        except requests.exceptions.ConnectionError:
            print("[OllamaBrain] WARNING: Could not connect to Ollama at localhost:11434")

    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe and then infer with Ollama."""
        text = self.stt_engine.transcribe(audio_data)
        if not text:
            return "Sorry, I couldn't hear that properly."

        print(f"[OllamaBrain] STT Text: {text}")
        return self.process_text(text)

    def process_text(self, text: str) -> str:
        """Run text through Ollama, handling multiple turns if tools are called."""
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        
        tools = self.tool_registry.get_openai_tools_list()
        
        # Maximum allowed tool execution turns to prevent infinite loops
        max_turns = 5
        
        for turn in range(max_turns):
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "tools": tools,
            }
            
            try:
                r = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
                if r.status_code != 200:
                    print(f"[OllamaBrain] Inference failed: {r.status_code} {r.text}")
                    return "Sorry, my brain stopped working properly."
                    
                data = r.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                # Append the assistant's message to the conversation
                messages.append(message)
                
                # If there are tool calls, execute them
                if "tool_calls" in message and message["tool_calls"]:
                    for tool_call in message["tool_calls"]:
                        fn = tool_call["function"]
                        name = fn["name"]
                        args = fn["arguments"]
                        
                        print(f"[OllamaBrain] Executing tool: {name} => {args}")
                        
                        # Pack into our dict structure that execute_parsed_tool_call expects
                        call_dict = {
                            "name": name,
                            "arguments": args
                        }
                        
                        result_str = self.tool_registry.execute_parsed_tool_call(call_dict)
                        print(f"[OllamaBrain] Tool result: {result_str}")
                        
                        # Append the tool result to the conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": name,
                            "content": result_str
                        })
                        
                    # Continue the loop so Ollama can analyze the tool results
                    continue
                else:
                    # Final text response
                    return message.get("content", "I did it, but I have nothing to say.")
                    
            except Exception as e:
                print(f"[OllamaBrain] Request failed: {e}")
                import traceback
                traceback.print_exc()
                return "Sorry, something went wrong with the connection."
                
        return "I tried to use tools too many times and gave up."
