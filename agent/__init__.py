"""
AI Sales Meeting Agent package.
"""

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.summarizer import summarize_meeting_transcript
from agent.transcriber import transcribe_meeting_audio
from agent.extractor_ollama import extract_crm_data as extract_crm_ollama
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient
from agent.live_listener import LiveMeetingListener, listen_to_live_meeting_and_summarize
from agent.meeting_joiner import MeetingJoiner, GoogleMeetJoiner
from agent.audio_devices import list_input_devices, recommend_input_device, find_loopback_device

__all__ = [
    "SalesMeetingAgent",
    "AgentResult",
    "summarize_meeting_transcript",
    "transcribe_meeting_audio",
    "extract_crm_ollama",
    "validate_extraction",
    "ValidationError",
    "MockCRMClient",
    "LiveMeetingListener",
    "listen_to_live_meeting_and_summarize",
    "MeetingJoiner",
    "GoogleMeetJoiner",
    "list_input_devices",
    "recommend_input_device",
    "find_loopback_device",
]
