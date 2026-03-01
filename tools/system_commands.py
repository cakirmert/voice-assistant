"""
System Commands — Built-in tools for the voice assistant.
"""
import subprocess
import datetime
import platform
import psutil
import os


def run_command(command: str) -> str:
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.expanduser("~"),
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\nError: {result.stderr.strip()}"
        return output if output else "(command completed with no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def get_system_info() -> str:
    """Get CPU, RAM, and disk usage information."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return (
        f"CPU: {cpu_percent}% used\n"
        f"RAM: {memory.percent}% used ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)\n"
        f"Disk: {disk.percent}% used ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)\n"
        f"OS: {platform.system()} {platform.release()}"
    )


def open_application(name: str) -> str:
    """Open an application by name (Windows)."""
    try:
        # Common app mappings
        app_map = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "terminal": "wt.exe",
            "task manager": "taskmgr.exe",
            "paint": "mspaint.exe",
            "browser": "start msedge",
            "edge": "start msedge",
            "chrome": "start chrome",
            "firefox": "start firefox",
            "spotify": "start spotify",
        }
        cmd = app_map.get(name.lower(), f"start {name}")
        subprocess.Popen(cmd, shell=True)
        return f"Opened {name}"
    except Exception as e:
        return f"Error opening {name}: {e}"


def control_volume(action: str) -> str:
    """Control system volume (up, down, mute) using Windows API keys via PowerShell."""
    try:
        action_lower = action.lower()
        if action_lower == "mute":
            # 173 = APPCOMMAND_VOLUME_MUTE
            cmd = "$obj = new-object -com wscript.shell; $obj.SendKeys([char]173)"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
            return "Toggled volume mute."
        elif action_lower == "down":
            # 174 = APPCOMMAND_VOLUME_DOWN
            cmd = "$obj = new-object -com wscript.shell; for($i=0; $i -lt 5; $i++){$obj.SendKeys([char]174)}"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
            return "Turned volume down."
        elif action_lower == "up":
            # 175 = APPCOMMAND_VOLUME_UP
            cmd = "$obj = new-object -com wscript.shell; for($i=0; $i -lt 5; $i++){$obj.SendKeys([char]175)}"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
            return "Turned volume up."
        else:
            return f"Invalid volume action: {action}. Use 'up', 'down', or 'mute'."
    except Exception as e:
        return f"Error controlling volume: {e}"


def register_system_tools(registry):
    """Register all system tools with the tool registry."""
    registry.register(
        name="run_command",
        description="Execute a shell command on the system and return its output. Use for checking files, running scripts, etc.",
        parameters={
            "command": {
                "description": "The shell command to execute",
                "type": "str",
            }
        },
        handler=run_command,
    )

    registry.register(
        name="get_current_time",
        description="Get the current date and time.",
        parameters={},
        handler=get_current_time,
    )

    registry.register(
        name="get_system_info",
        description="Get system information including CPU usage, RAM usage, and disk space.",
        parameters={},
        handler=get_system_info,
    )

    registry.register(
        name="open_application",
        description="Open an application by name (e.g., notepad, calculator, browser, terminal, spotify).",
        parameters={
            "name": {
                "description": "Name of the application to open",
                "type": "str",
            }
        },
        handler=open_application,
    )

    registry.register(
        name="control_volume",
        description="Control the system audio volume. Actions allowed: 'up', 'down', 'mute'.",
        parameters={
            "action": {
                "description": "The volume action to perform: 'up', 'down', or 'mute'",
                "type": "str",
            }
        },
        handler=control_volume,
    )
