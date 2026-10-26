"""
Mock CRM client -- simulates Salesforce's Contact/Account/Opportunity API
using a local JSON file as the "database". Same duplicate-detection logic
and same write_crm_record() interface as salesforce_client.py, so you can
swap between them freely once real Salesforce access is sorted out.

No authentication, no network calls, no setup required. Good for demos,
testing the extraction pipeline end-to-end, and satisfying the "CRM
Integration Implementation" requirement without needing live credentials.

Usage:
    from agent.mock_crm_client import MockCRMClient
    crm = MockCRMClient()
    result = crm.write_crm_record(extracted_json)
"""

import json
import re
import uuid
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mock_crm_db.json"

EMPTY_DB = {"Contact": {}, "Account": {}, "Opportunity": {}, "Task": {}}

# How similar two names need to be (0-100) to be treated as the same
# person/company when there's no exact email/name match to go on.
CONTACT_FUZZY_THRESHOLD = 88
ACCOUNT_FUZZY_THRESHOLD = 90

_COMPANY_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|corp|co|company|group|industries)\b\.?", re.IGNORECASE)


def _normalize_company(name: str) -> str:
    name = _COMPANY_SUFFIXES.sub("", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _company_similarity(a: str, b: str) -> float:
    """
    Combines two comparisons to be robust to both extra corporate suffixes
    ('Inc', 'LLC') and inconsistent spacing ('GrowthTech' vs 'Growth Tech').
    """
    na, nb = _normalize_company(a), _normalize_company(b)
    spaced_score = fuzz.token_sort_ratio(na, nb)
    despaced_score = fuzz.ratio(na.replace(" ", ""), nb.replace(" ", ""))
    return max(spaced_score, despaced_score)


class MockCRMClient:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._load()

    def _load(self):
        if self.db_path.exists() and self.db_path.stat().st_size > 0:
            try:
                with open(self.db_path) as f:
                    self.db = json.load(f)
                # Ensure all tables exist
                for table in EMPTY_DB:
                    if table not in self.db:
                        self.db[table] = {}
            except Exception:
                self.db = {k: dict(v) for k, v in EMPTY_DB.items()}
                self._save()
        else:
            self.db = {k: dict(v) for k, v in EMPTY_DB.items()}
            self._save()


    def _save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.db, f, indent=2)

    def _new_id(self, prefix):
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    # -----------------------------------------------------------------
    # Duplicate detection & Lookup
    # -----------------------------------------------------------------

    def find_contact_by_email(self, email):
        if not email:
            return None
        for cid, rec in self.db.get("Contact", {}).items():
            if rec.get("Email") and rec.get("Email").strip().lower() == email.strip().lower():
                return cid
        return None

    def find_contact_fuzzy(self, first_name, last_name):
        full_name = f"{first_name or ''} {last_name or ''}".strip()
        if not full_name:
            return None

        best_id, best_score = None, 0
        for cid, rec in self.db.get("Contact", {}).items():
            existing_name = f"{rec.get('FirstName') or ''} {rec.get('LastName') or ''}".strip()
            if not existing_name:
                continue
            score = fuzz.token_sort_ratio(full_name, existing_name)
            if score > best_score:
                best_id, best_score = cid, score

        if best_score >= CONTACT_FUZZY_THRESHOLD:
            return best_id
        return None

    def find_account_by_name(self, name):
        if not name:
            return None
        for aid, rec in self.db.get("Account", {}).items():
            if rec.get("Name") and rec.get("Name").strip().lower() == name.strip().lower():
                return aid
        return None

    def find_account_fuzzy(self, name):
        if not name:
            return None

        best_id, best_score = None, 0
        for aid, rec in self.db.get("Account", {}).items():
            existing_name = rec.get("Name") or ""
            if not existing_name:
                continue
            score = _company_similarity(name, existing_name)
            if score > best_score:
                best_id, best_score = aid, score

        if best_score >= ACCOUNT_FUZZY_THRESHOLD:
            return best_id
        return None

    def find_opportunity(self, account_id, opp_name=None):
        """Finds matching opportunity by AccountId and optional Name similarity."""
        if not account_id:
            return None
        for oid, rec in self.db.get("Opportunity", {}).items():
            if rec.get("AccountId") == account_id:
                if not opp_name:
                    return oid
                score = fuzz.token_sort_ratio(opp_name, rec.get("Name", ""))
                if score >= 70:
                    return oid
        return None

    def find_task(self, account_id, subject):
        """Finds existing task with matching subject for the account."""
        if not subject:
            return None
        for tid, rec in self.db.get("Task", {}).items():
            if rec.get("AccountId") == account_id:
                score = fuzz.token_sort_ratio(subject, rec.get("Subject", ""))
                if score >= 80:
                    return tid
        return None

    # -----------------------------------------------------------------
    # Record creation & updates
    # -----------------------------------------------------------------

    def create_record(self, sobject_type, fields):
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        prefix = sobject_type[:3].upper() if sobject_type != "Task" else "TSK"
        record_id = self._new_id(prefix)
        if sobject_type not in self.db:
            self.db[sobject_type] = {}
        self.db[sobject_type][record_id] = clean_fields
        self._save()
        return record_id

    def update_record(self, sobject_type, record_id, fields):
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        if record_id in self.db.get(sobject_type, {}):
            self.db[sobject_type][record_id].update(clean_fields)
            self._save()
            return record_id
        return None

    def write_crm_record(self, extraction: dict, actions: Optional[dict] = None) -> dict:
        """
        Executes automated CRM record management:
        1. Determines create vs update for Account (with fuzzy matching)
        2. Determines create vs update for Contact (email or fuzzy matching)
        3. Determines create vs update for Opportunity
        4. Determines create vs update for Action Items / Tasks
        """
        contact_data = extraction.get("Contact", {}) or {}
        account_data = extraction.get("Account", {}) or {}
        opp_data = extraction.get("Opportunity", {}) or {}

        summary = {
            "account_id": None, "account_action": None,
            "contact_id": None, "contact_action": None,
            "opportunity_id": None, "opportunity_action": None,
            "tasks_created": 0, "tasks_updated": 0, "task_ids": []
        }

        # 1. Account: Create or Update
        account_name = account_data.get("Name")
        existing_account_id = self.find_account_by_name(account_name)
        if not existing_account_id:
            existing_account_id = self.find_account_fuzzy(account_name)
            if existing_account_id:
                summary["account_action"] = "updated_fuzzy"

        account_fields = {
            "Name": account_data.get("Name"),
            "Industry": account_data.get("Industry"),
            "NumberOfEmployees": account_data.get("NumberOfEmployees"),
            "Website": account_data.get("Website"),
            "BudgetRange__c": account_data.get("BudgetRange__c"),
            "PainPoints__c": account_data.get("PainPoints__c"),
        }

        if existing_account_id:
            self.update_record("Account", existing_account_id, account_fields)
            summary["account_id"] = existing_account_id
            summary["account_action"] = summary["account_action"] or "updated"
        elif account_name:
            account_id = self.create_record("Account", account_fields)
            summary["account_id"] = account_id
            summary["account_action"] = "created"

        # 2. Contact: Create or Update
        email = contact_data.get("Email")
        existing_contact_id = self.find_contact_by_email(email)
        if not existing_contact_id:
            existing_contact_id = self.find_contact_fuzzy(
                contact_data.get("FirstName"), contact_data.get("LastName"))
            if existing_contact_id:
                summary["contact_action"] = "updated_fuzzy"

        contact_fields = {
            "FirstName": contact_data.get("FirstName"),
            "LastName": contact_data.get("LastName") or "Unknown",
            "Title": contact_data.get("Title"),
            "Department": contact_data.get("Department"),
            "Email": contact_data.get("Email"),
            "Phone": contact_data.get("Phone"),
            "AccountId": summary["account_id"],
        }

        if existing_contact_id:
            self.update_record("Contact", existing_contact_id, contact_fields)
            summary["contact_id"] = existing_contact_id
            summary["contact_action"] = summary["contact_action"] or "updated"
        elif contact_data.get("FirstName") or contact_data.get("LastName"):
            contact_id = self.create_record("Contact", contact_fields)
            summary["contact_id"] = contact_id
            summary["contact_action"] = "created"

        # 3. Opportunity: Create or Update
        if opp_data.get("Name") and summary["account_id"]:
            opp_fields = {
                "Name": opp_data.get("Name"),
                "AccountId": summary["account_id"],
                "Amount": opp_data.get("Amount"),
                "StageName": opp_data.get("StageName") or "Prospecting",
                "CloseDate": (date.today() + timedelta(days=90)).isoformat(),
                "NextStep": opp_data.get("NextStep"),
                "Competitor__c": opp_data.get("Competitor__c"),
                "Description": opp_data.get("Description"),
            }
            existing_opp_id = self.find_opportunity(summary["account_id"], opp_data.get("Name"))
            if existing_opp_id:
                self.update_record("Opportunity", existing_opp_id, opp_fields)
                summary["opportunity_id"] = existing_opp_id
                summary["opportunity_action"] = "updated"
            else:
                opp_id = self.create_record("Opportunity", opp_fields)
                summary["opportunity_id"] = opp_id
                summary["opportunity_action"] = "created"
        elif opp_data.get("Name") and not summary["account_id"]:
            summary["opportunity_action"] = "skipped_no_account"

        # 4. Action Items & Tasks: Create or Update
        task_candidates = []
        if actions:
            # Add general sales tasks
            for t in actions.get("tasks", []):
                task_candidates.append({
                    "title": t.get("title"),
                    "priority": t.get("priority", "High"),
                    "owner": t.get("suggested_owner", "Sales Rep"),
                    "due": t.get("due_timeline", "Immediate"),
                    "category": "Sales Action"
                })
            # Add Next Best Action required follow-ups
            nba = actions.get("next_best_action", {})
            for fu in nba.get("required_follow_ups", []):
                task_candidates.append({
                    "title": fu.get("action"),
                    "priority": "High" if "immediate" in fu.get("category", "").lower() else "Medium",
                    "owner": fu.get("suggested_owner", "Sales Rep"),
                    "due": fu.get("deadline", "Next milestone"),
                    "category": fu.get("category", "Follow-up")
                })

        for tc in task_candidates:
            subj = tc.get("title")
            if not subj:
                continue
            existing_task_id = self.find_task(summary["account_id"], subj)
            task_fields = {
                "Subject": subj,
                "Priority": tc.get("priority", "Medium"),
                "Owner": tc.get("owner", "Sales Rep"),
                "ActivityDate": tc.get("due"),
                "Category": tc.get("category"),
                "Status": "Open",
                "AccountId": summary["account_id"],
                "WhoId": summary["contact_id"],
            }
            if existing_task_id:
                self.update_record("Task", existing_task_id, task_fields)
                summary["tasks_updated"] += 1
                summary["task_ids"].append(existing_task_id)
            else:
                tid = self.create_record("Task", task_fields)
                summary["tasks_created"] += 1
                summary["task_ids"].append(tid)

        return summary

    # -----------------------------------------------------------------
    # Inspection helpers -- handy for demos
    # -----------------------------------------------------------------

    def dump(self):
        """Print the whole mock CRM database -- useful for showing in a demo."""
        print(json.dumps(self.db, indent=2))

    def reset(self):
        """Wipe the mock CRM back to empty. Useful between demo runs."""
        self.db = {k: dict(v) for k, v in EMPTY_DB.items()}
        self._save()


if __name__ == "__main__":
    crm = MockCRMClient()
    crm.reset()

    example = {
        "Contact": {"FirstName": "Sarah", "LastName": "Johnson", "Title": "Marketing Director",
                     "Department": "Marketing", "Email": "sarah.johnson@growthtech.com", "Phone": None},
        "Account": {"Name": "GrowthTech", "Industry": "SaaS", "NumberOfEmployees": 50,
                     "Website": None, "BudgetRange__c": "$30K annually", "PainPoints__c": None},
        "Opportunity": {"Name": "Marketing Automation - GrowthTech", "Amount": 30000,
                          "StageName": "Demo Scheduled", "NextStep": "Demo on Friday",
                          "Competitor__c": "HubSpot", "Description": "Interested in marketing automation."},
    }
    result = crm.write_crm_record(example)
    print("Write result:", json.dumps(result, indent=2))
    print("\nFull mock CRM contents:")
    crm.dump()

    print("\n--- Writing the SAME contact again to test duplicate detection ---")
    result2 = crm.write_crm_record(example)
    print("Second write result:", json.dumps(result2, indent=2))
    assert result2["contact_action"] == "reused_existing", "Duplicate detection failed!"
    assert result2["account_action"] == "reused_existing", "Duplicate detection failed!"
    print("\nDuplicate detection confirmed working: existing records were reused, not duplicated.")
