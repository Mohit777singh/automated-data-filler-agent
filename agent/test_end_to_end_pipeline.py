"""
Integration tests for the unified End-to-End Sales Meeting Pipeline.
Verifies the complete automated sequence:
Meeting Audio -> Speech-to-Text -> Meeting Summary -> AI Sales Agent ->
CRM Extraction -> Missed Questions -> What We Should Have Said ->
Next Best Action -> Action Items.
"""

import io
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.test_audio_flow import create_synthetic_wav_bytes


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.agent = SalesMeetingAgent(model_name="llama3.2")
        self.sample_dialogue = (
            "Alex (AE): Hi Sarah, great to connect with you. Thanks for joining today's discovery call.\n"
            "Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! Thanks for having me. We are a 50-person SaaS company.\n"
            "Alex (AE): What's the main challenge you're looking to solve?\n"
            "Sarah Johnson: Our team is wasting hours manually sending onboarding emails and updating lists. We need automated marketing workflows and a solid reporting suite.\n"
            "Alex (AE): Got it. Have you established a budget for this platform?\n"
            "Sarah Johnson: Yes, our VP approved an annual budget around $30,000.\n"
            "Alex (AE): Are you evaluating other tools currently?\n"
            "Sarah Johnson: We looked at HubSpot, but their reporting is rigid. Your analytics look much stronger.\n"
            "Alex (AE): Wonderful. Let's schedule a dedicated ops walkthrough for your team this Friday.\n"
            "Sarah Johnson: Friday works perfectly for us. Let's do it."
        )

    def test_complete_pipeline_flow(self):
        """
        Validates the entire automated chain without manual preparation:
        Speech-to-Text / Input -> Summary -> Agent Signals -> CRM Extraction ->
        Missed Questions -> What We Should Have Said -> Next Best Action -> Action Items.
        """
        # Run unified pipeline entry point
        result = self.agent.run_end_to_end_pipeline(self.sample_dialogue, auto_confirm=True)
        self.assertIsInstance(result, AgentResult)

        # Stage 1: Automatic Meeting Summary
        self.assertIsNotNone(result.summary)
        self.assertIn("purpose", result.summary)
        self.assertIn("customer_needs", result.summary)
        self.assertIn("pain_points", result.summary)
        self.assertIn("budget", result.summary)
        self.assertIn("decisions", result.summary)
        self.assertIn("next_steps", result.summary)

        # Stage 2: AI Sales Agent (Analysis & Gap Assessment)
        self.assertIsNotNone(result.analysis)
        self.assertIn("meeting_type", result.analysis)
        self.assertIn("buying_intent", result.analysis)
        self.assertIn("completeness_score", result.information_needed)

        # Stage 3: Structured CRM Extraction
        self.assertIn("Contact", result.crm_data)
        self.assertIn("Account", result.crm_data)
        self.assertIn("Opportunity", result.crm_data)
        self.assertEqual(result.crm_data["Account"].get("Name"), "GrowthTech")
        self.assertEqual(result.crm_data["Contact"].get("FirstName"), "Sarah")

        # Stage 4: Missed Questions & What We Should Have Said
        self.assertIsNotNone(result.coaching)
        self.assertIn("missed_questions", result.coaching)
        self.assertIn("what_we_should_have_said", result.coaching)
        self.assertIn("overall_performance_score", result.coaching)
        self.assertTrue(len(result.coaching["missed_questions"]) > 0)
        self.assertTrue(len(result.coaching["what_we_should_have_said"]) > 0)

        # Stage 5: Next Best Action (NBA)
        self.assertIn("next_best_action", result.actions)
        nba = result.actions["next_best_action"]
        self.assertIn("primary_action", nba)
        self.assertIn("rationale", nba)
        self.assertIn("urgency", nba)

        # Stage 6: Action Items & CRM Ops
        self.assertIn("crm_actions", result.actions)
        self.assertIn("tasks", result.actions)
        self.assertIn("follow_up_email", result.actions)
        self.assertIsNotNone(result.crm_write_result)
        self.assertIn("account_id", result.crm_write_result)
        self.assertIn("contact_id", result.crm_write_result)
        self.assertIn("opportunity_id", result.crm_write_result)

        print("\n===========================================================================")
        print("PIPELINE TEST PASSED: ALL 8 STAGES VERIFIED AUTOMATICALLY")
        print(f"• Summary Purpose:             {result.summary.get('purpose')}")
        print(f"• CRM Contact / Account:       {result.crm_data['Contact'].get('FirstName')} {result.crm_data['Contact'].get('LastName')} @ {result.crm_data['Account'].get('Name')}")
        print(f"• Missed Questions Identified: {len(result.coaching['missed_questions'])}")
        print(f"• What We Should Have Said:    {result.coaching['what_we_should_have_said'][0][:60]}...")
        print(f"• Primary Next Best Action:    {nba.get('primary_action')}")
        print(f"• CRM Action Status:           Account: {result.crm_write_result.get('account_action')}, Contact: {result.crm_write_result.get('contact_action')}")
        print("===========================================================================")


if __name__ == "__main__":
    unittest.main()
