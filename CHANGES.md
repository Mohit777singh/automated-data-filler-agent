# Debug Report: Live Meeting Audio Capture → CRM Pipeline

## Root cause (two separate bugs, both fixed)

### Bug 1 — the bot never captured real audio
`GoogleMeetJoiner` (`agent/meeting_joiner.py`) only drives Selenium to open
Chrome and join the URL. It never touches audio at all. Its only capture
mechanism was scraping Google Meet's on-screen **live captions** out of the
DOM using a single hardcoded CSS selector set
(`div[jsname='YSxPfc'], div.yg51vc, span.VfPpkd-vQzf8d`). Google changes
these class/jsname values periodically, and captions have to be manually
enabled and stay enabled — if either breaks, `captured_transcript_lines`
stays permanently empty, silently.

Separately, `LiveMeetingListener` (`agent/live_listener.py`) *does* do real
audio capture + Whisper transcription — but it opens the **default
microphone**. On macOS, a browser tab's audio is never routed into any
input device by the OS. So even this path would only capture whatever the
room's physical mic overheard, not what the meeting participants said.

### Bug 2 — failures were being masked with fake data
Throughout `app.py`, every real-capture code path had a silent fallback:
```python
if not transcribed_text:
    st.warning("No speech detected... Loaded sample transcript.")
    transcribed_text = SAMPLE_LIVE_DIALOGUE   # "Alex... Sarah Johnson..."
...
run_agent_triggered = True   # proceeds to CRM extraction anyway
```
and `GoogleMeetJoiner.leave_and_summarize(fallback_sample_if_empty=...)`
did the same. So when real capture failed (which it always did, per Bug
1), the app quietly substituted the canned "Alex/Sarah Johnson/GrowthTech"
script and ran the *full* CRM pipeline on it — producing a plausible-looking
but entirely fictional CRM report. This is why you saw "CRM extraction and
report generation" appearing to sort of work even though the transcript
was empty.

A third instance of the same bug was in the Zoom/Meet ingestion tab: pasting
a bare `meet.google.com` URL silently substituted `SAMPLE_MEET_TRANSCRIPT`
and ran the pipeline as if it had ingested something real.

## What was fixed

1. **`agent/audio_devices.py` (new)** — cross-platform input-device
   enumeration, loopback/virtual-device detection (BlackHole, VB-Cable,
   Stereo Mix, PulseAudio monitor, etc.), a `recommend_input_device()`
   helper, and the shared `logger` used by every pipeline stage.

2. **`agent/live_listener.py`** — rewritten with:
   - Stage logging: `LISTENER: starting…`, `AUDIO STREAM: detected`,
     per-chunk `TRANSCRIPTION: started/…`, `TRANSCRIPT: generated segment…`.
   - A `diagnostics` dict (`first_callback_received`, `total_callbacks`,
     `chunks_transcribed`, `chunks_skipped_silence`, `last_error`) exposed
     to callers instead of being invisible.
   - A watchdog thread that logs a loud warning if zero audio callbacks
     arrive within 5s (wrong device / permission denied).
   - Device validation in `start()` — raises a clear error listing valid
     device indices instead of an opaque PortAudio failure.
   - `process_and_summarize()` still raises `ValueError` on an empty
     transcript, now with diagnostics attached — no fallback text.

3. **`agent/meeting_joiner.py`** — rewritten with:
   - Stage logging for every step: navigated, name filled, mic/cam muted,
     join button clicked, **join confirmed** (new: waits for a "Leave
     call" control to actually verify the bot is in the room, not just
     that a button was clicked), captions hotkey sent, caption stream
     detected/matched selector, per-line capture.
   - Expanded, order-tried list of caption CSS selectors (Google's DOM is
     not a stable public API) plus `aria-live` regions, which are more
     stable than class names.
   - A loud warning if 20s pass with zero caption matches.
   - `leave_and_summarize()` **no longer accepts a fallback-sample
     parameter**. If nothing was captured it raises `RuntimeError` with
     the exact message the task asked for — *"Meeting joined successfully
     but audio stream is unavailable"* — plus diagnostics, instead of
     silently substituting sample dialogue.

4. **`app.py`**:
   - Removed all silent `transcribed_text = SAMPLE_LIVE_DIALOGUE` /
     `SAMPLE_MEET_TRANSCRIPT` fallbacks in the three real-capture code
     paths (Join Live Meeting, Live Audio Listener Option 1 & 2, Zoom/Meet
     URL paste). Each now shows the literal error message
     **"Meeting joined successfully but audio stream is unavailable."**
     and does **not** run the CRM pipeline.
   - "Load Sample" buttons remain, but are relabeled "Load **Demo** Sample"
     and explicitly tell the user it's not real meeting data.
   - New loopback-device banner: shows whether a BlackHole/VB-Cable-style
     device was detected; if not, renders macOS setup instructions
     (`agent/audio_devices.MACOS_BLACKHOLE_SETUP_MD`) inline, and
     auto-selects the loopback device in the dropdown when one exists.
   - New **"🪵 Pipeline log"** expander on both the Join-Live-Meeting and
     Live-Audio-Listener tabs, streaming the stage log lines described
     above directly into the UI.

5. **Ollama / CRM extraction** (`agent/extractor_ollama.py`,
   `agent/sales_meeting_agent.py`) — already correctly gated: CRM
   extraction only runs after a non-empty transcript
   (`process_meeting`/`process_meeting_audio` raise `ValueError` on empty
   input) and already surfaces a friendly error if the Ollama app isn't
   running. No behavioral change needed there beyond the upstream fixes
   above that stop fake transcripts from ever reaching it.

## macOS audio setup (required for real meeting-audio capture)

Browser tab audio is never automatically exposed to Python. Install
**BlackHole** and route your system/Chrome output through a Multi-Output
Device so the meeting's audio reaches an input device this app can record
from. Full step-by-step instructions are shown automatically in the app's
"Live Audio Listener" tab when no loopback device is detected, and are also
in `agent/audio_devices.py` (`MACOS_BLACKHOLE_SETUP_MD`).

## Recommended real capture path

The DOM-caption approach (`GoogleMeetJoiner`) is inherently fragile because
it depends on Google's unstable internal DOM. For production reliability,
prefer: **Live Audio Listener tab → select your BlackHole (or other
loopback) device → Start Live Host Listening**. This uses the real
Meeting Audio → Whisper → Transcript → Ollama → CRM pipeline end to end,
with full stage logging visible in the Pipeline Log panel.
