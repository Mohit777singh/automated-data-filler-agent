"""
Audio device detection & macOS system-audio (loopback) helper utilities.

WHY THIS FILE EXISTS
---------------------
On macOS (and to a lesser extent Windows/Linux), a normal microphone input
device can ONLY capture sound physically picked up by the mic. When a bot
"joins" a Google Meet / Zoom call in a Chrome browser, the other
participants' voices are played out through the Mac's speakers/Chrome tab
audio -- they are NOT automatically routed into any Python-visible input
device. Selenium launching Chrome does not grant Python access to that
audio stream; there is no OS-level API that hands "the audio a browser
tab is playing" to an arbitrary process without a virtual audio driver.

To capture what's being SAID IN THE MEETING (not just what the bot's own
mic overhears from the room), the machine needs a virtual/loopback audio
device such as BlackHole (macOS), which lets you route Chrome's output
audio into an input device that `sounddevice` can then record from.

This module:
  1. Enumerates available input devices via `sounddevice`.
  2. Heuristically flags devices that look like a loopback / virtual
     audio device (BlackHole, Loopback, Soundflower, Stereo Mix, VB-Cable).
  3. Exposes setup instructions the UI can render.
  4. Provides a single shared logger used across the audio/transcription
     pipeline so every stage prints a clear, timestamped line.
"""

import logging
import platform
import sys
from typing import Any, Dict, List, Optional

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared pipeline logger
# ---------------------------------------------------------------------------
# One logger, used by meeting_joiner.py, live_listener.py, transcriber.py,
# extractor_ollama.py and app.py so every stage of
#   Meeting Audio -> Audio Capture -> Speech-to-Text -> Transcript ->
#   Ollama Llama -> CRM extraction
# emits a consistent, greppable log line. app.py also attaches a
# StreamHandler that writes into a Streamlit-visible buffer (see app.py).
logger = logging.getLogger("sales_meeting_agent.pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.propagate = False


# Substrings that indicate a device is a virtual/loopback device rather
# than a physical microphone.
_LOOPBACK_HINTS = [
    "blackhole",      # macOS - https://existential.audio/blackhole/
    "loopback",       # macOS (Rogue Amoeba Loopback) / generic
    "soundflower",    # macOS legacy
    "stereo mix",     # Windows
    "vb-audio",       # Windows (VB-Cable)
    "vb-cable",
    "cable output",   # VB-Cable alt name
    "voicemeeter",    # Windows
    "monitor of",     # Linux/PulseAudio monitor source
    "pulse monitor",
]


def is_macos() -> bool:
    return platform.system() == "Darwin"


def list_input_devices() -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {index, name, max_input_channels,
    default_samplerate, is_loopback}. Empty list if sounddevice/portaudio
    isn't available (logged, never raises).
    """
    devices: List[Dict[str, Any]] = []
    if not SOUNDDEVICE_AVAILABLE:
        logger.warning("AUDIO DEVICES: sounddevice/PortAudio is not available on this system.")
        return devices

    try:
        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", f"Device {idx}")
                name_lower = name.lower()
                # Verify that this device can actually open a 16kHz audio stream
                try:
                    sd.check_input_settings(device=idx, channels=1, dtype="float32", samplerate=16000)
                except Exception:
                    continue

                devices.append({
                    "index": idx,
                    "name": name,
                    "max_input_channels": d.get("max_input_channels", 0),
                    "default_samplerate": d.get("default_samplerate", 44100),
                    "is_loopback": any(hint in name_lower for hint in _LOOPBACK_HINTS),
                })
    except Exception as e:
        logger.error(f"AUDIO DEVICES: failed to query devices: {e}")

    return devices


def find_loopback_device() -> Optional[Dict[str, Any]]:
    """Returns the first device that looks like a virtual/loopback audio
    device (e.g. BlackHole), or None if none is installed/found."""
    for d in list_input_devices():
        if d["is_loopback"]:
            return d
    return None


def recommend_input_device() -> Dict[str, Any]:
    """
    Decides what device the app should default to for capturing MEETING
    audio (as opposed to the bot operator's own voice), and explains why.
    """
    devices = list_input_devices()
    loopback = find_loopback_device()

    if loopback:
        return {
            "recommended_index": loopback["index"],
            "recommended_name": loopback["name"],
            "reason": (
                f"Found virtual audio device '{loopback['name']}'. Route your "
                "meeting app's output to it (see setup instructions) and select "
                "it here to capture what OTHER participants say."
            ),
            "loopback_available": True,
            "devices": devices,
        }

    return {
        "recommended_index": None,
        "recommended_name": None,
        "reason": (
            "No virtual/loopback audio device detected. Without one, this "
            "machine can only record whatever your physical microphone "
            "picks up from the room -- it will NOT reliably capture "
            "other participants' voices from a Google Meet / Zoom call. "
            "Install BlackHole (macOS) or VB-Cable (Windows) for real "
            "meeting-audio capture -- see setup instructions below."
        ),
        "loopback_available": False,
        "devices": devices,
    }


MACOS_BLACKHOLE_SETUP_MD = """
### macOS setup: capture real meeting audio with BlackHole

A browser tab's audio never reaches Python by itself. BlackHole is a free
virtual audio driver that lets you route what Chrome is playing (the Meet/
Zoom call) into an input device this app can record from.

1. **Install BlackHole (2ch)**
   ```
   brew install blackhole-2ch
   ```
   or download the installer from https://existential.audio/blackhole/

2. **Create a Multi-Output Device** (so you can still hear the meeting
   yourself while it's captured):
   - Open **Audio MIDI Setup** (Cmd+Space -> "Audio MIDI Setup").
   - Click **+** (bottom-left) -> **Create Multi-Output Device**.
   - Check both **BlackHole 2ch** and your normal output (e.g. MacBook
     Speakers) in the list.
   - Rename it to something like "Meeting + BlackHole".

3. **Set Chrome's / macOS's output to the Multi-Output Device**
   - System Settings -> Sound -> Output -> select "Meeting + BlackHole".
   - (Or use a per-app audio router like Loopback/SoundSource if you only
     want Chrome routed, not the whole system.)

4. **In this app**, open the "Live Audio Listener" tab and pick
   **BlackHole 2ch** as the Audio Input Device. Click "Test Mic (3s)"
   while the meeting is playing -- you should see a nonzero peak volume.

5. Join your Google Meet / Zoom call as normal, unmute participants,
   and click **Start Live Host Listening Now**. The audio the meeting
   is playing (via BlackHole) is now what gets recorded and transcribed
   -- not your physical mic.

**Windows equivalent:** install VB-Audio Virtual Cable
(https://vb-audio.com/Cable/), set it as your playback device, and select
"CABLE Output" as the input device here.
"""
