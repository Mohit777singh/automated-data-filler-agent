"""
Live Meeting Listener & Real-Time Audio Transcription Engine.
Listens to live meeting audio from a chosen input device (microphone OR a
virtual/loopback device such as BlackHole that carries the meeting's
speaker output -- see agent/audio_devices.py), transcribes segments using
Whisper, and automatically generates comprehensive meeting summaries and
CRM actions when the meeting concludes.

Logging checkpoints:
    - "Audio stream detected" (first callback received)
    - "Transcribing audio..." (chunk passed to whisper)
    - "Transcript generated" (transcription completed)
"""

import time
import queue
import threading
import numpy as np
from typing import Optional, Callable, List, Dict, Any

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

from agent.transcriber import transcribe_meeting_audio
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.audio_devices import logger, list_input_devices

_NO_CALLBACK_WARNING_SEC = 5.0


class LiveMeetingListener:
    """
    Listens to live sales meetings in real time, generates streaming transcripts,
    and automatically forwards the complete meeting transcript to the Sales Meeting Agent.
    """

    @staticmethod
    def _device_mismatch_warning(selected_device: Optional[int], devices: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Returns a warning when the user selected the wrong input while a loopback device exists."""
        if devices is None:
            devices = list_input_devices()

        if selected_device is None:
            return None

        loopback = next((d for d in devices if d.get("is_loopback")), None)
        if not loopback:
            return None

        if selected_device == loopback.get("index"):
            return None

        selected = next((d for d in devices if d.get("index") == selected_device), None)
        selected_name = selected.get("name") if selected else f"device {selected_device}"
        loopback_name = loopback.get("name", "the loopback device")
        return (
            f"Selected input '{selected_name}' is a physical microphone or local mic, not the meeting audio stream. "
            f"When a virtual device like '{loopback_name}' is available, use it instead to capture the actual meeting conversation; "
            "otherwise the transcript will mismatch what is being said in the call."
        )

    def __init__(
        self,
        model_size: str = "small",
        sample_rate: int = 16000,
        chunk_duration_sec: float = 5.0,
        energy_threshold: float = 0.002,
        on_segment_callback: Optional[Callable[[str], None]] = None,
        agent: Optional[SalesMeetingAgent] = None,
    ):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.energy_threshold = energy_threshold
        self.on_segment_callback = on_segment_callback
        self.agent = agent or SalesMeetingAgent()

        self._is_listening = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._transcribe_thread: Optional[threading.Thread] = None
        self._audio_stream: Optional[Any] = None

        self.transcript_segments: List[Dict[str, Any]] = []
        self.accumulated_transcript: str = ""
        self._all_session_audio: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0

        self.diagnostics: Dict[str, Any] = {
            "device_index": None,
            "device_name": None,
            "first_callback_received": False,
            "total_callbacks": 0,
            "total_samples_received": 0,
            "chunks_transcribed": 0,
            "chunks_skipped_silence": 0,
            "last_error": None,
        }

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for incoming audio buffers."""
        if status:
            logger.warning(f"AUDIO STREAM status: {status}")

        if not self._is_listening:
            return

        if not self.diagnostics["first_callback_received"]:
            self.diagnostics["first_callback_received"] = True
            logger.info("Audio stream detected")

        self.diagnostics["total_callbacks"] += 1
        self.diagnostics["total_samples_received"] += len(indata)

        if indata.ndim > 1:
            mono_data = indata.mean(axis=1)
        else:
            mono_data = indata.flatten()
        mono_float = mono_data.copy().astype(np.float32)
        self._all_session_audio.append(mono_float)
        self._audio_queue.put(mono_float)

    def _transcription_worker(self):
        """Worker thread that processes accumulated audio chunks and runs Whisper."""
        buffer = []
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_sec)

        while self._is_listening or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.25)
                buffer.append(data)
            except queue.Empty:
                continue

            current_samples = sum(len(b) for b in buffer)
            if current_samples >= samples_per_chunk or (not self._is_listening and buffer):
                full_chunk = np.concatenate(buffer)
                buffer = []

                # Compute RMS energy for voice activity
                rms = np.sqrt(np.mean(full_chunk ** 2))
                if rms < self.energy_threshold and self._is_listening:
                    self.diagnostics["chunks_skipped_silence"] += 1
                    continue

                try:
                    res = transcribe_meeting_audio(full_chunk, model_size=self.model_size)
                    if not res.get("success"):
                        self.diagnostics["last_error"] = res.get("error")
                        continue

                    text = (res.get("transcript") or "").strip()
                    if text and len(text) > 1:
                        self.diagnostics["chunks_transcribed"] += 1
                        elapsed_sec = int(time.time() - self._start_time)
                        mins, secs = divmod(elapsed_sec, 60)
                        timestamp = f"{mins:02d}:{secs:02d}"

                        segment_info = {
                            "timestamp": timestamp,
                            "text": text,
                        }

                        with self._lock:
                            self.transcript_segments.append(segment_info)
                            if self.accumulated_transcript:
                                self.accumulated_transcript += f"\n[{timestamp}] {text}"
                            else:
                                self.accumulated_transcript = f"[{timestamp}] {text}"

                        if self.on_segment_callback:
                            try:
                                self.on_segment_callback(f"[{timestamp}] {text}")
                            except Exception as cb_err:
                                logger.warning(f"TRANSCRIPT callback error: {cb_err}")
                except Exception as e:
                    self.diagnostics["last_error"] = str(e)
                    logger.error(f"TRANSCRIPTION error: {e}")

    def _watchdog(self):
        """Warns if the audio device never fires a single callback."""
        time.sleep(_NO_CALLBACK_WARNING_SEC)
        if self._is_listening and not self.diagnostics["first_callback_received"]:
            logger.warning(
                f"Audio stream not detected from device '{self.diagnostics.get('device_name')}'. "
                f"Please ensure microphone permissions are granted or BlackHole loopback is enabled."
            )

    def start(self, device: Optional[int] = None) -> bool:
        """
        Starts listening to live audio from the given input device index.
        """
        if self._is_listening:
            return True

        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Live audio recording requires 'sounddevice' (PortAudio). "
                "Install it with: pip install sounddevice"
            )

        device_name = "System Default"
        devices = list_input_devices()
        if device is not None:
            match = next((d for d in devices if d["index"] == device), None)
            if match is None:
                available = ", ".join(f"[{d['index']}] {d['name']}" for d in devices) or "(none found)"
                raise RuntimeError(
                    f"Audio input device index {device} was not found. Available devices: {available}"
                )
            device_name = match["name"]

        warning = self._device_mismatch_warning(device, devices)
        if warning:
            logger.warning(warning)

        self.diagnostics = {
            "device_index": device,
            "device_name": device_name,
            "first_callback_received": False,
            "total_callbacks": 0,
            "total_samples_received": 0,
            "chunks_transcribed": 0,
            "chunks_skipped_silence": 0,
            "last_error": None,
        }

        self._is_listening = True
        self._start_time = time.time()
        self.transcript_segments = []
        self.accumulated_transcript = ""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start transcription worker thread
        self._transcribe_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._transcribe_thread.start()

        # Start watchdog thread
        threading.Thread(target=self._watchdog, daemon=True).start()

        # Start sounddevice InputStream
        try:
            self._audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.5)
            )
            self._audio_stream.start()
            return True
        except Exception as e:
            self._is_listening = False
            if self._transcribe_thread and self._transcribe_thread.is_alive():
                self._transcribe_thread.join(timeout=1.0)
            raise RuntimeError(
                f"Could not open live audio stream on device '{device_name}': {e}. "
                f"Please ensure microphone permissions are granted."
            )

    def stop(self, wait_for_drain: bool = True) -> str:
        """
        Stops listening and returns the complete accumulated transcript.
        """
        if not self._is_listening:
            return self.accumulated_transcript

        self._is_listening = False

        if self._audio_stream:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            self._audio_stream = None

        if wait_for_drain and self._transcribe_thread:
            self._transcribe_thread.join(timeout=3.0)

        # Unified full-session pass: gives Whisper the full conversation context without cut-off words
        if self._all_session_audio:
            try:
                full_audio = np.concatenate(self._all_session_audio)
                if len(full_audio) >= self.sample_rate:
                    res = transcribe_meeting_audio(full_audio, model_size=self.model_size)
                    final_txt = (res.get("transcript") or "").strip()
                    if final_txt and len(final_txt) > 5:
                        with self._lock:
                            self.accumulated_transcript = final_txt
            except Exception as e:
                logger.warning(f"Full-session transcription pass notice: {e}")

        return self.accumulated_transcript

    def feed_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Feeds an external audio buffer for testing."""
        if audio_data.ndim > 1:
            mono = audio_data.mean(axis=1)
        else:
            mono = audio_data.flatten()
        self._audio_queue.put(mono.astype(np.float32))

    def get_live_transcript(self) -> str:
        """Returns the transcript accumulated so far."""
        with self._lock:
            return self.accumulated_transcript

    def process_and_summarize(self, auto_confirm: bool = True) -> AgentResult:
        """
        Finalizes the live listening session and routes transcript into CRM extraction.
        """
        transcript = self.stop()
        if not transcript.strip():
            raise RuntimeError(
                "Meeting joined successfully but no audio stream detected. "
                "Please enable BlackHole audio routing."
            )

        return self.agent.run_end_to_end_pipeline(transcript, auto_confirm=auto_confirm)


def listen_to_live_meeting_and_summarize(
    duration_seconds: float = 10.0,
    agent: Optional[SalesMeetingAgent] = None,
    auto_confirm: bool = True,
    device: Optional[int] = None,
) -> AgentResult:
    """
    Convenience function that listens to live meeting audio for a specified duration
    and automatically produces the complete meeting summary and CRM actions.
    """
    listener = LiveMeetingListener(agent=agent)
    listener.start(device=device)
    time.sleep(duration_seconds)
    return listener.process_and_summarize(auto_confirm=auto_confirm)
