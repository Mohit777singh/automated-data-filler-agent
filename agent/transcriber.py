"""
Audio Transcription & Speech-to-Text Module for Live Sales Meetings.
Uses faster-whisper (or OpenAI Whisper fallback) to convert audio buffers/streams
into high-accuracy text transcripts.
"""

import io
import os
import wave
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Union, BinaryIO, Optional, List
import numpy as np

logger = logging.getLogger("sales_meeting_agent.pipeline")

_FASTER_WHISPER_CACHE: Dict[str, Any] = {}
_OPENAI_WHISPER_CACHE: Dict[str, Any] = {}


def get_faster_whisper_model(model_size: str = "base"):
    """
    Loads and caches the faster-whisper model (optimized for CPU/GPU).
    """
    global _FASTER_WHISPER_CACHE
    if model_size not in _FASTER_WHISPER_CACHE:
        try:
            from faster_whisper import WhisperModel
            # 'int8' or 'float32' for fast CPU inference
            _FASTER_WHISPER_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            logger.warning(f"Could not load faster-whisper '{model_size}': {e}")
            return None
    return _FASTER_WHISPER_CACHE.get(model_size)


def get_openai_whisper_model(model_size: str = "base"):
    """
    Fallback loader for openai-whisper if available.
    """
    global _OPENAI_WHISPER_CACHE
    if model_size not in _OPENAI_WHISPER_CACHE:
        try:
            import whisper
            _OPENAI_WHISPER_CACHE[model_size] = whisper.load_model(model_size)
        except Exception as e:
            logger.warning(f"Could not load openai-whisper '{model_size}': {e}")
            return None
    return _OPENAI_WHISPER_CACHE.get(model_size)


def _load_audio_to_numpy(audio_source: Union[str, Path, bytes, BinaryIO, np.ndarray]) -> np.ndarray:
    """
    Decodes audio from numpy arrays, file paths, raw bytes, or stream buffers into a 16kHz mono float32 numpy array.
    """
    target_sr = 16000

    # 1. Direct numpy array input
    if isinstance(audio_source, np.ndarray):
        audio = audio_source.copy()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_val = np.iinfo(audio.dtype).max
                audio = (audio / max_val).astype(np.float32)
            else:
                audio = audio.astype(np.float32)
        # Auto-Gain Control: boost quiet speech signals to clear acoustic levels
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0002:
            audio = (audio / peak * 0.7).astype(np.float32)
        return audio.astype(np.float32)

    # Extract raw bytes if stream or bytes-like object
    raw_bytes = None
    if isinstance(audio_source, (bytes, bytearray)):
        raw_bytes = bytes(audio_source)
    elif hasattr(audio_source, "read"):
        raw_bytes = audio_source.read()
        if hasattr(audio_source, "seek"):
            try:
                audio_source.seek(0)
            except Exception:
                pass

    # 2. Try soundfile first
    try:
        import soundfile as sf
        if raw_bytes is not None:
            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        else:
            audio, sr = sf.read(str(audio_source), dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != target_sr:
            indices = np.round(np.arange(0, len(audio), sr / target_sr)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        return audio.astype(np.float32)
    except Exception:
        pass

    # 3. Try standard wave module
    try:
        if raw_bytes is not None:
            wav_stream = io.BytesIO(raw_bytes)
        else:
            wav_stream = open(str(audio_source), "rb")

        with wave.open(wav_stream, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)

            if sampwidth == 2:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            if framerate != target_sr:
                indices = np.round(np.arange(0, len(audio), framerate / target_sr)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]

            return audio.astype(np.float32)
    except Exception:
        pass

    raise ValueError("Could not decode audio format. Expected 16kHz float32 array or standard audio stream.")


def transcribe_meeting_audio(
    audio_source: Union[str, Path, bytes, BinaryIO, np.ndarray],
    model_size: str = "base",
    language: Optional[str] = "en"
) -> Dict[str, Any]:
    """
    Transcribes audio chunks/recordings into formatted text using faster-whisper or whisper.
    
    Returns:
        Dict containing:
        - transcript: str
        - language: str
        - segments: list
        - success: bool
    """
    try:
        logger.info("Transcribing audio...")
        audio_np = _load_audio_to_numpy(audio_source)

        # Silence / white noise filter: prevent Whisper from hallucinating on silence
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        if rms < 0.0008:
            return {
                "transcript": "",
                "language": language or "en",
                "segments": [],
                "success": True,
            }
        fw_model = get_faster_whisper_model(model_size=model_size)
        if fw_model is not None:
            segments_gen, info = fw_model.transcribe(
                audio_np,
                language=language or "en",
                beam_size=1,
                vad_filter=True,
            )
            segments_list = []
            text_parts = []
            for seg in segments_gen:
                text_parts.append(seg.text.strip())
                segments_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })
            transcript_text = " ".join(text_parts).strip()
            if transcript_text:
                logger.info("Transcript generated")
            return {
                "transcript": transcript_text,
                "language": info.language if info else (language or "en"),
                "segments": segments_list,
                "success": True,
            }

        # 2. Fallback to OpenAI Whisper
        ow_model = get_openai_whisper_model(model_size=model_size)
        if ow_model is not None:
            options = {
                "fp16": False,
                "language": language or "en",
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "initial_prompt": "B2B sales meeting discussion about CRM architecture, SaaS platforms, pricing, ROI, integrations, enterprise workflows, onboarding, customer success, technical implementation, company size, budget, decision makers, and next steps.",
                "no_speech_threshold": 0.5,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
            }
            result = ow_model.transcribe(audio_np, **options)
            transcript_text = (result.get("text") or "").strip()
            if transcript_text:
                logger.info("Transcript generated")
            return {
                "transcript": transcript_text,
                "language": result.get("language", language or "en"),
                "segments": result.get("segments", []),
                "success": True,
            }

        raise RuntimeError("No Whisper engine (faster-whisper or whisper) is available.")

    except Exception as e:
        logger.warning(f"Transcription error: {e}")
        return {
            "transcript": "",
            "language": language or "en",
            "segments": [],
            "success": False,
            "error": str(e),
        }
