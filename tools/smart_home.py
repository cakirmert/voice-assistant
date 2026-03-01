import logging

logger = logging.getLogger("smart_home")

def control_brightness(level: int) -> str:
    """Set the brightness of smart lights. level: 0-100."""
    try:
        # High-level simulation of Matter/Smart home control
        # In a real scenario, this would call a CLI like 'chip-tool' or an API
        logger.info(f"[SmartHome] Setting brightness to {level}%")
        return f"Brightness has been adjusted to {level}%."
    except Exception as e:
        return f"Error controlling brightness: {e}"

def smart_device_action(device: str, action: str, value: str = "") -> str:
    """Perform an action on a smart device (e.g. 'turn on', 'set color')."""
    try:
        logger.info(f"[SmartHome] Device: {device}, Action: {action}, Value: {value}")
        return f"The {device} has been told to {action} {value}."
    except Exception as e:
        return f"Error on smart device {device}: {e}"

def register_smart_home_tools(registry):
    """Register smart home tools."""
    registry.register(
        name="control_brightness",
        description="Adjust the brightness of the room lights or a specific lamp.",
        parameters={
            "level": {
                "description": "Brightness percentage (0 to 100)",
                "type": "int",
            }
        },
        handler=control_brightness,
    )
    registry.register(
        name="smart_device_action",
        description="Control smart home devices like switches, lights, or plugs.",
        parameters={
            "device": {"description": "The name of the device", "type": "str"},
            "action": {"description": "The action to take (on, off, color, etc)", "type": "str"},
            "value": {"description": "Optional value for the action", "type": "str"}
        },
        handler=smart_device_action,
    )
