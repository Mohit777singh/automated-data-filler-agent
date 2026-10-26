"""
Test suite for automated CRM create/update decisions across
Contact, Account, Opportunity, and Action Items.
"""

import sys
import unittest
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.mock_crm_client import MockCRMClient
from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult


class TestCRMAutoOperations(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_db.close()
        self.crm = MockCRMClient(db_path=self.temp_db.name)
        self.agent = SalesMeetingAgent(model_name="llama3.2", crm_client=self.crm)

    def tearDown(self):
        p = Path(self.temp_db.name)
        if p.exists():
            p.unlink()

    def test_auto_create_and_update_cycle(self):
        """
        Tests that the first meeting creates new records (Account, Contact, Opportunity, Action Items),
        and a subsequent meeting updates the existing records instead of duplicating them.
        """
        extraction_1 = {
            "Contact": {
                "FirstName": "Sarah",
                "LastName": "Johnson",
                "Title": "Marketing Director",
                "Email": "sarah.johnson@growthtech.com",
                "Department": "Marketing"
            },
            "Account": {
                "Name": "GrowthTech",
                "Industry": "SaaS",
                "NumberOfEmployees": 50,
                "BudgetRange__c": "$30,000"
            },
            "Opportunity": {
                "Name": "Marketing Workflows - GrowthTech",
                "Amount": 30000,
                "StageName": "Demo Scheduled",
                "NextStep": "Dedicated ops walkthrough on Friday"
            }
        }
        actions_1 = {
            "tasks": [
                {"title": "Execute Friday Demo", "priority": "High", "suggested_owner": "Account Executive", "due_timeline": "Friday"}
            ],
            "next_best_action": {
                "required_follow_ups": [
                    {"category": "Immediate Customer Outreach", "action": "Send recap email", "suggested_owner": "AE", "deadline": "2 hours"}
                ]
            }
        }

        # 1. First execution -> should CREATE records
        res_1 = self.crm.write_crm_record(extraction_1, actions=actions_1)
        self.assertEqual(res_1["account_action"], "created")
        self.assertEqual(res_1["contact_action"], "created")
        self.assertEqual(res_1["opportunity_action"], "created")
        self.assertGreaterEqual(res_1["tasks_created"], 2)
        acc_id = res_1["account_id"]
        con_id = res_1["contact_id"]
        opp_id = res_1["opportunity_id"]

        # 2. Second execution with same company (slight variation: 'Growth Tech') & contact -> should UPDATE / REUSE
        extraction_2 = {
            "Contact": {
                "FirstName": "Sarah",
                "LastName": "Johnson",
                "Title": "VP of Marketing",  # Updated title
                "Email": "sarah.johnson@growthtech.com"
            },
            "Account": {
                "Name": "Growth Tech Inc",  # Fuzzy match
                "NumberOfEmployees": 65,     # Updated size
                "BudgetRange__c": "$35,000"
            },
            "Opportunity": {
                "Name": "Marketing Workflows - GrowthTech",
                "Amount": 35000,             # Updated value
                "StageName": "Proposal / Price Quote",
                "NextStep": "Send contract"
            }
        }
        actions_2 = {
            "tasks": [
                {"title": "Execute Friday Demo", "priority": "Low", "suggested_owner": "Account Executive", "due_timeline": "Completed"}
            ]
        }

        res_2 = self.crm.write_crm_record(extraction_2, actions=actions_2)
        # Verify Account reuse / update
        self.assertEqual(res_2["account_id"], acc_id)
        self.assertIn("update", res_2["account_action"])
        self.assertEqual(self.crm.db["Account"][acc_id]["NumberOfEmployees"], 65)

        # Verify Contact reuse / update
        self.assertEqual(res_2["contact_id"], con_id)
        self.assertIn("update", res_2["contact_action"])
        self.assertEqual(self.crm.db["Contact"][con_id]["Title"], "VP of Marketing")

        # Verify Opportunity update
        self.assertEqual(res_2["opportunity_id"], opp_id)
        self.assertEqual(res_2["opportunity_action"], "updated")
        self.assertEqual(self.crm.db["Opportunity"][opp_id]["StageName"], "Proposal / Price Quote")
        self.assertEqual(self.crm.db["Opportunity"][opp_id]["Amount"], 35000)

        # Verify Task update
        self.assertGreaterEqual(res_2["tasks_updated"], 1)

        print("\n[SUCCESS] Automated CRM create vs update cycle verified for Contact, Account, Opportunity, and Tasks!")

    def test_agent_automatic_crm_pipeline_execution(self):
        """
        Tests that SalesMeetingAgent.run_end_to_end_pipeline automatically performs
        the complete CRM write with no manual steps required.
        """
        transcript = (
            "Alex (AE): Hi Sarah, great connecting with you today.\n"
            "Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! We are a 50-person SaaS company looking for workflow automation.\n"
            "Alex (AE): What budget has been allocated?\n"
            "Sarah Johnson: Our VP approved $30,000 annually.\n"
            "Alex (AE): Let's schedule a dedicated ops walkthrough for your team this Friday.\n"
            "Sarah Johnson: Sounds great."
        )

        result = self.agent.run_end_to_end_pipeline(transcript, auto_confirm=True)
        self.assertIsInstance(result, AgentResult)
        self.assertIsNotNone(result.crm_write_result)
        self.assertIn("account_id", result.crm_write_result)
        self.assertIn("contact_id", result.crm_write_result)
        self.assertIn("opportunity_id", result.crm_write_result)
        self.assertGreaterEqual(len(result.crm_write_result.get("task_ids", [])), 1)
        print("\n[SUCCESS] Agent automatically performed CRM work after meeting analysis!")


if __name__ == "__main__":
    unittest.main()
