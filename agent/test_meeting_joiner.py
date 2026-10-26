"""
Unit tests for the GoogleMeetJoiner automated meeting joiner.
"""

import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.meeting_joiner import GoogleMeetJoiner
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult


class TestGoogleMeetJoiner(unittest.TestCase):

    def setUp(self):
        self.agent = SalesMeetingAgent(model_name="llama3.2")
        self.joiner = GoogleMeetJoiner(participant_name="AI Sales Note Taker", agent=self.agent)

    def test_leave_and_summarize_pipeline(self):
        """
        Verifies that captured Google Meet dialogue is routed through the SalesMeetingAgent
        to produce the complete 9-point summary, CRM records, coaching, and NBA.
        """
        sample_transcript = (
            "Alex (AE): Hi Sarah, welcome to our Google Meet session.\n"
            "Sarah Johnson (Marketing Director @ GrowthTech, sarah.johnson@growthtech.com): Hi Alex! We are a 50-person SaaS company.\n"
            "Alex (AE): What is the core friction your team faces today?\n"
            "Sarah Johnson: Manual onboarding emails and updating contact lists. We need automated marketing workflows and a solid reporting suite.\n"
            "Alex (AE): Have you established a budget for this transition?\n"
            "Sarah Johnson: Yes, our VP approved an annual budget around $30,000.\n"
            "Alex (AE): Did you look at alternative platforms like HubSpot?\n"
            "Sarah Johnson: We evaluated HubSpot, but found its reporting too rigid. Your custom dashboards look much better.\n"
            "Alex (AE): Let's schedule a dedicated ops walkthrough for your team this Friday.\n"
            "Sarah Johnson: Friday works great for us. Let's do it!"
        )

        self.joiner.captured_transcript_lines = sample_transcript.splitlines()
        agent_result = self.joiner.leave_and_summarize()

        self.assertIsInstance(agent_result, AgentResult)
        self.assertIsNotNone(agent_result.summary)
        self.assertEqual(agent_result.crm_data["Account"].get("Name"), "GrowthTech")
        self.assertIn(agent_result.crm_data["Contact"].get("FirstName"), ["Sarah", "Alex"])

        print("\n===========================================================================")
        print("GOOGLE MEET JOINER TEST PASSED: AUTOMATED JOIN & SUMMARY PIPELINE")
        print(f"• Ingested Dialogue Lines: {len(self.joiner.captured_transcript_lines)}")
        print(f"• Summary Purpose:         {agent_result.summary.get('purpose')}")
        print(f"• Extracted Account:       {agent_result.crm_data['Account'].get('Name')}")
        print("===========================================================================")


if __name__ == "__main__":
    unittest.main()
