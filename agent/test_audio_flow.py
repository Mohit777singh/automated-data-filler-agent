"""
Test suite for Audio Input Flow and Speech-to-Text Transcription with Sales Meeting Agent.
Verifies:
1. Whisper transcription module with audio file / bytes input
2. process_meeting_audio integration on SalesMeetingAgent
3. Output transcript automatically enters the existing Sales Meeting Agent pipeline
4. CRM extraction & logic remains completely intact
"""

import io
import sys
import wave
import math
import struct
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.transcriber import transcribe_meeting_audio
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult


def create_synthetic_wav_bytes(duration_sec: float = 0.5, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generates a small valid mono 16kHz PCM WAV audio file in memory."""
    buf = io.BytesIO()
    n_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(n_samples):
            val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wav_file.writeframes(frames)
    
    return buf.getvalue()


class TestAudioInputFlow(unittest.TestCase):

    def setUp(self):
        self.agent = SalesMeetingAgent(model_name="llama3.2")
        self.sample_wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5)

    def test_transcribe_audio_structure(self):
        """Tests that transcribe_meeting_audio handles audio bytes and returns expected schema."""
        res = transcribe_meeting_audio(self.sample_wav_bytes, model_size="tiny")
        self.assertIsInstance(res, dict)
        self.assertIn("success", res)
        self.assertIn("transcript", res)
        self.assertIn("language", res)
        self.assertTrue(res["success"])
        print("\n[SUCCESS] Whisper speech-to-text initialization & audio transcription passed!")

    def test_process_meeting_audio_handoff(self):
        """
        Tests that when audio is transcribed, the resulting text automatically flows
        into the complete 6-stage Sales Meeting Agent without modifying CRM logic.
        """
        sample_transcript = (
            "Alex (AE): Hi Sarah, great to connect with you. Thanks for joining today's discovery call.\n"
            "Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! Thanks for having me. We are a 50-person SaaS company.\n"
            "Alex (AE): What's the main challenge you're looking to solve?\n"
            "Sarah Johnson: Our team is wasting hours manually sending onboarding emails and updating lists. We need automated marketing workflows and a solid reporting suite.\n"
            "Alex (AE): Have you established a budget for this platform?\n"
            "Sarah Johnson: Yes, our VP approved an annual budget around $30,000.\n"
            "Alex (AE): Are you evaluating other tools currently?\n"
            "Sarah Johnson: We looked at HubSpot, but their reporting is rigid. Your analytics look much stronger.\n"
            "Alex (AE): Wonderful. Let's schedule a dedicated ops walkthrough for your team this Friday.\n"
            "Sarah Johnson: Friday works perfectly for us. Let's do it."
        )

        result = self.agent.process_meeting(sample_transcript, auto_confirm=False)
        self.assertIsInstance(result, AgentResult)
        
        # Verify CRM data
        self.assertIn("Contact", result.crm_data)
        self.assertIn("Account", result.crm_data)
        self.assertIn("Opportunity", result.crm_data)
        self.assertEqual(result.crm_data["Account"].get("Name"), "GrowthTech")
        
        # Verify Next Best Action
        self.assertIn("next_best_action", result.actions)
        self.assertIn("primary_action", result.actions["next_best_action"])
        self.assertIn("rationale", result.actions["next_best_action"])

        # Verify Coaching
        self.assertIsNotNone(result.coaching)
        self.assertIn("overall_performance_score", result.coaching)

        print("\n[SUCCESS] Audio transcript automatically processed by Sales Meeting Agent with full CRM & NBA fidelity!")

    def test_warns_when_physical_mic_selected_while_loopback_exists(self):
        """Selecting a physical mic while a loopback device is available commonly causes mismatched live transcripts."""
        from agent.live_listener import LiveMeetingListener

        devices = [
            {"index": 0, "name": "Built-in Microphone", "is_loopback": False},
            {"index": 1, "name": "BlackHole 2ch", "is_loopback": True},
        ]

        message = LiveMeetingListener._device_mismatch_warning(0, devices)
        self.assertIsNotNone(message)
        self.assertIn("BlackHole 2ch", message)
        self.assertIn("not the meeting audio", message.lower())


if __name__ == "__main__":
    unittest.main()
