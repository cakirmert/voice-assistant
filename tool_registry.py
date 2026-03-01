"""
Tool Registry — Framework for registering and executing tools that Phi-4 can call.
"""
import json
import subprocess
import traceback
from typing import Callable, Any


class Tool:
    """Represents a callable tool with metadata for Phi-4's tool-calling format."""

    def __init__(self, name: str, description: str, parameters: dict, handler: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_openai_dict(self) -> dict:
        """Convert to OpenAI standard tool definition format."""
        
        # Ensure parameters has valid structure for OpenAI
        properties = {}
        required = []
        for param_name, param_details in self.parameters.items():
            properties[param_name] = {
                "type": param_details.get("type", "string"),
                "description": param_details.get("description", "")
            }
            # As a simple heuristic, we mark all parameters as required
            required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments, return result as string."""
        try:
            result = self.handler(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {e}"


class ToolRegistry:
    """Registry that manages available tools and generates Phi-4 tool definitions."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict, handler: Callable):
        """Register a new tool."""
        self.tools[name] = Tool(name, description, parameters, handler)

    def get_openai_tools_list(self) -> list[dict]:
        """Generate list of all tool definitions in standard OpenAI format."""
        return [tool.to_openai_dict() for tool in self.tools.values()]

    def execute_tool_call(self, tool_call_str: str) -> str:
        """
        Parse and execute a tool call from JSON string.
        """
        try:
            call = json.loads(tool_call_str)
            return self.execute_parsed_tool_call(call)
        except json.JSONDecodeError:
            return f"Failed to parse tool call: {tool_call_str}"
        except Exception as e:
            return f"Tool execution error: {e}\n{traceback.format_exc()}"

    def execute_parsed_tool_call(self, call: dict) -> str:
        """Execute a tool call that is already parsed into a dictionary."""
        try:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            # Sometimes tools like OpenAI return arguments as a JSON string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return f"Failed to parse tool arguments JSON: {arguments}"

            if tool_name not in self.tools:
                return f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"

            return self.tools[tool_name].execute(**arguments)
        except Exception as e:
            return f"Tool execution error: {e}\n{traceback.format_exc()}"

    def parse_tool_calls(self, response: str) -> list[dict]:
        """
        Extract tool calls from Phi-4's response.
        Phi-4 outputs tool calls as JSON objects.
        """
        calls = []
        # Try to find JSON objects in the response
        # Phi-4 typically outputs clean JSON for tool calls
        response = response.strip()
        try:
            # Single tool call
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "name" in parsed:
                calls.append(parsed)
            elif isinstance(parsed, list):
                calls.extend([c for c in parsed if isinstance(c, dict) and "name" in c])
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            import re
            json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
            matches = re.findall(json_pattern, response)
            for match in matches:
                try:
                    calls.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
        return calls
