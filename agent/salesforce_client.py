"""
Salesforce integration layer for the Automated Data Filler Agent.

Handles:
1. OAuth authentication (username-password flow) -> access token
2. Duplicate detection (check if Contact/Account already exists before creating)
3. Writing validated extraction output into Contact, Account, Opportunity records

Setup: put your 6 Salesforce credentials in a `.env` file (see .env.example):
    SF_USERNAME=you@yourorg.com
    SF_PASSWORD=your_password
    SF_SECURITY_TOKEN=your_security_token
    SF_INSTANCE_URL=https://yourdomain-dev-ed.develop.my.salesforce.com
    SF_CLIENT_ID=your_consumer_key
    SF_CLIENT_SECRET=your_consumer_secret

Usage:
    from agent.salesforce_client import SalesforceClient
    sf = SalesforceClient()
    sf.write_crm_record(extracted_json)
"""

import os
import re
import requests
from rapidfuzz import fuzz

from agent.http_utils import request_with_retry, FriendlyRequestError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LOGIN_URL = "https://login.salesforce.com/services/oauth2/token"
API_VERSION = "v60.0"
CONTACT_FUZZY_THRESHOLD = 88
ACCOUNT_FUZZY_THRESHOLD = 90

_COMPANY_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|corp|co|company|group|industries)\b\.?", re.IGNORECASE)


def _normalize_company(name: str) -> str:
    name = _COMPANY_SUFFIXES.sub("", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def _company_similarity(a: str, b: str) -> float:
    """Same logic as mock_crm_client.py -- see EVALUATOR_PREP.md for testing/reasoning."""
    na, nb = _normalize_company(a), _normalize_company(b)
    spaced_score = fuzz.token_sort_ratio(na, nb)
    despaced_score = fuzz.ratio(na.replace(" ", ""), nb.replace(" ", ""))
    return max(spaced_score, despaced_score)


class SalesforceClient:
    def __init__(self):
        self.username = os.environ.get("SF_USERNAME")
        self.password = os.environ.get("SF_PASSWORD")
        self.security_token = os.environ.get("SF_SECURITY_TOKEN")
        self.client_id = os.environ.get("SF_CLIENT_ID")
        self.client_secret = os.environ.get("SF_CLIENT_SECRET")

        missing = [k for k, v in {
            "SF_USERNAME": self.username, "SF_PASSWORD": self.password,
            "SF_SECURITY_TOKEN": self.security_token, "SF_CLIENT_ID": self.client_id,
            "SF_CLIENT_SECRET": self.client_secret,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Missing Salesforce credentials in .env: {missing}")

        self.access_token = None
        self.instance_url = None
        self._authenticate()

    def _authenticate(self):
        """OAuth username-password flow -> access token + real instance URL."""
        payload = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password + self.security_token,
        }
        try:
            resp = request_with_retry(
                lambda: requests.post(LOGIN_URL, data=payload, timeout=30),
                friendly_name="Salesforce login",
            )
        except FriendlyRequestError as e:
            raise RuntimeError(str(e))

        if resp.status_code != 200:
            raise RuntimeError(f"Salesforce auth failed: {resp.status_code} {resp.text}")

        data = resp.json()
        self.access_token = data["access_token"]
        self.instance_url = data["instance_url"]  # Salesforce returns the correct one
        print(f"Authenticated. Connected to: {self.instance_url}")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _api_url(self, path):
        return f"{self.instance_url}/services/data/{API_VERSION}/{path}"

    # -----------------------------------------------------------------
    # Duplicate detection
    # -----------------------------------------------------------------

    def find_contact_by_email(self, email):
        if not email:
            return None
        query = f"SELECT Id, FirstName, LastName FROM Contact WHERE Email = '{email}' LIMIT 1"
        resp = requests.get(self._api_url("query"), headers=self._headers(),
                              params={"q": query}, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        return records[0]["Id"] if records else None

    def find_account_by_name(self, name):
        if not name:
            return None
        safe_name = name.replace("'", "\\'")
        query = f"SELECT Id, Name FROM Account WHERE Name = '{safe_name}' LIMIT 1"
        resp = requests.get(self._api_url("query"), headers=self._headers(),
                              params={"q": query}, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        return records[0]["Id"] if records else None

    def find_contact_fuzzy(self, first_name, last_name):
        """
        Fallback when there's no email to match on. Pulls a candidate set
        via SOQL LIKE on last name (cheap server-side pre-filter), then
        ranks candidates with rapidfuzz to confirm a real match rather than
        just a substring hit.
        """
        full_name = f"{first_name or ''} {last_name or ''}".strip()
        if not last_name:
            return None

        safe_last = last_name.replace("'", "\\'")
        query = f"SELECT Id, FirstName, LastName FROM Contact WHERE LastName LIKE '%{safe_last}%' LIMIT 50"
        resp = requests.get(self._api_url("query"), headers=self._headers(),
                              params={"q": query}, timeout=30)
        resp.raise_for_status()
        candidates = resp.json().get("records", [])

        best_id, best_score = None, 0
        for rec in candidates:
            candidate_name = f"{rec.get('FirstName') or ''} {rec.get('LastName') or ''}".strip()
            score = fuzz.token_sort_ratio(full_name, candidate_name)
            if score > best_score:
                best_id, best_score = rec["Id"], score

        return best_id if best_score >= CONTACT_FUZZY_THRESHOLD else None

    def find_account_fuzzy(self, name):
        """Fallback fuzzy match for company names via SOQL LIKE pre-filter + rapidfuzz ranking."""
        if not name:
            return None

        # Use the first word as a cheap server-side pre-filter (e.g. "GrowthTech" from "GrowthTech Inc")
        first_word = name.split()[0].replace("'", "\\'")
        query = f"SELECT Id, Name FROM Account WHERE Name LIKE '%{first_word}%' LIMIT 50"
        resp = requests.get(self._api_url("query"), headers=self._headers(),
                              params={"q": query}, timeout=30)
        resp.raise_for_status()
        candidates = resp.json().get("records", [])

        best_id, best_score = None, 0
        for rec in candidates:
            score = _company_similarity(name, rec.get("Name") or "")
            if score > best_score:
                best_id, best_score = rec["Id"], score

        return best_id if best_score >= ACCOUNT_FUZZY_THRESHOLD else None

    # -----------------------------------------------------------------
    # Record creation
    # -----------------------------------------------------------------

    def create_record(self, sobject_type, fields):
        clean_fields = {k: v for k, v in fields.items() if v is not None}
        try:
            resp = request_with_retry(
                lambda: requests.post(self._api_url(f"sobjects/{sobject_type}"),
                                        headers=self._headers(), json=clean_fields, timeout=30),
                friendly_name=f"Salesforce ({sobject_type} creation)",
            )
        except FriendlyRequestError as e:
            raise RuntimeError(
                f"CRM temporarily unreachable while creating {sobject_type} -- "
                f"your extraction was not lost, try writing to the CRM again. ({e})"
            )

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create {sobject_type}: {resp.status_code} {resp.text}")
        return resp.json()["id"]

    def write_crm_record(self, extraction: dict) -> dict:
        """
        Takes the output of extract_crm_data() (already validated) and writes
        it to Salesforce. Checks for existing Account/Contact first to avoid
        duplicates. Returns a summary dict of what was created/reused.
        """
        contact_data = extraction.get("Contact", {})
        account_data = extraction.get("Account", {})
        opp_data = extraction.get("Opportunity", {})

        summary = {"account_id": None, "account_action": None,
                    "contact_id": None, "contact_action": None,
                    "opportunity_id": None, "opportunity_action": None}

        # --- Account (create or reuse) ---
        account_name = account_data.get("Name")
        existing_account_id = self.find_account_by_name(account_name)
        if not existing_account_id:
            existing_account_id = self.find_account_fuzzy(account_name)
            if existing_account_id:
                summary["account_action"] = "reused_existing_fuzzy"
        if existing_account_id:
            summary["account_id"] = existing_account_id
            summary["account_action"] = summary["account_action"] or "reused_existing"
        elif account_name:
            account_id = self.create_record("Account", {
                "Name": account_data.get("Name"),
                "Industry": account_data.get("Industry"),
                "NumberOfEmployees": account_data.get("NumberOfEmployees"),
                "Website": account_data.get("Website"),
            })
            summary["account_id"] = account_id
            summary["account_action"] = "created"

        # --- Contact (create or reuse, linked to Account) ---
        email = contact_data.get("Email")
        existing_contact_id = self.find_contact_by_email(email)
        if not existing_contact_id:
            existing_contact_id = self.find_contact_fuzzy(
                contact_data.get("FirstName"), contact_data.get("LastName"))
            if existing_contact_id:
                summary["contact_action"] = "reused_existing_fuzzy"
        if existing_contact_id:
            summary["contact_id"] = existing_contact_id
            summary["contact_action"] = summary["contact_action"] or "reused_existing"
        elif contact_data.get("FirstName") or contact_data.get("LastName"):
            contact_id = self.create_record("Contact", {
                "FirstName": contact_data.get("FirstName"),
                "LastName": contact_data.get("LastName") or "Unknown",  # LastName is required by Salesforce
                "Title": contact_data.get("Title"),
                "Department": contact_data.get("Department"),
                "Email": contact_data.get("Email"),
                "Phone": contact_data.get("Phone"),
                "AccountId": summary["account_id"],
            })
            summary["contact_id"] = contact_id
            summary["contact_action"] = "created"

        # --- Opportunity (always created if there's a name, linked to Account) ---
        if opp_data.get("Name") and summary["account_id"]:
            opp_id = self.create_record("Opportunity", {
                "Name": opp_data.get("Name"),
                "AccountId": summary["account_id"],
                "Amount": opp_data.get("Amount"),
                "StageName": opp_data.get("StageName") or "Prospecting",
                "CloseDate": _default_close_date(),
                "Description": opp_data.get("Description"),
            })
            summary["opportunity_id"] = opp_id
            summary["opportunity_action"] = "created"
        elif opp_data.get("Name") and not summary["account_id"]:
            summary["opportunity_action"] = "skipped_no_account"

        return summary


def _default_close_date():
    """Salesforce requires a CloseDate on every Opportunity. Defaults to 90 days out."""
    from datetime import date, timedelta
    return (date.today() + timedelta(days=90)).isoformat()


if __name__ == "__main__":
    # Quick manual test -- requires real credentials in .env
    example = {
        "Contact": {"FirstName": "Sarah", "LastName": "Johnson", "Title": "Marketing Director",
                     "Department": "Marketing", "Email": "sarah.johnson.test@growthtech-demo.com",
                     "Phone": None},
        "Account": {"Name": "GrowthTech Demo Co", "Industry": "SaaS",
                     "NumberOfEmployees": 50, "Website": None},
        "Opportunity": {"Name": "Marketing Automation - GrowthTech Demo Co",
                          "Amount": 30000, "StageName": "Prospecting",
                          "Description": "Interested in marketing automation."},
    }
    sf = SalesforceClient()
    result = sf.write_crm_record(example)
    print(result)
