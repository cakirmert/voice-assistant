import subprocess
import os

def search_spotify(query: str) -> str:
    """Search for a song or artist on Spotify and play it."""
    try:
        # Use the spotify:search: URI scheme to trigger a search
        # Replacing spaces with %20
        formatted_query = query.replace(" ", "%20")
        cmd = f"start spotify:search:{formatted_query}"
        subprocess.run(cmd, shell=True)
        return f"Searching Spotify for: {query}"
    except Exception as e:
        return f"Error searching Spotify: {e}"

def spotify_control(action: str) -> str:
    """Control Spotify playback (play, pause, next, back)."""
    # Simple shell trigger for basic control if Spotify is active
    # Note: Complex control usually requires Web API, but we can try basic media keys
    try:
        action = action.lower()
        # Mapping to media keys via powershell
        mapping = {
            "play": "[char]179", 
            "pause": "[char]179",
            "next": "[char]176",
            "back": "[char]177"
        }
        if action in mapping:
            cmd = f"$obj = new-object -com wscript.shell; $obj.SendKeys({mapping[action]})"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
            return f"Spotify {action} command sent."
        return f"Unknown spotify action: {action}"
    except Exception as e:
        return f"Error controlling Spotify: {e}"

def register_media_tools(registry):
    """Register media tools."""
    registry.register(
        name="search_spotify",
        description="Search for a song, artist, or album on Spotify and start playback.",
        parameters={
            "query": {
                "description": "The search term (e.g. 'Blinding Lights by The Weeknd')",
                "type": "str",
            }
        },
        handler=search_spotify,
    )
    registry.register(
        name="spotify_control",
        description="Control Spotify playback like play, pause, next, or previous track.",
        parameters={
            "action": {
                "description": "The action to perform: 'play', 'pause', 'next', 'back'",
                "type": "str",
            }
        },
        handler=spotify_control,
    )
