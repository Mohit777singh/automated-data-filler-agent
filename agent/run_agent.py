"""
Full end-to-end Automated Data Filler Agent flow, matching the
conversational demo in the problem statement:

  Agent: "Please paste your meeting summary, and I'll extract the CRM
          data automatically."
  User:  <pastes summary>
  Agent: "Here's what I extracted... confirm? (y/n)"
  User:  y
  Agent: "Created Contact, Account, Opportunity in the CRM."

Run from the project root:
    python -m agent.run_agent
"""

import json

from agent.extractor_ollama import extract_crm_data
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient


def print_extraction_summary(data: dict):
    c, a, o = data["Contact"], data["Account"], data["Opportunity"]
    print("\nHere's what I extracted:\n")
    print("Contact:")
    print(f"  Name: {c.get('FirstName') or ''} {c.get('LastName') or ''}".strip() or "  (none found)")
    if c.get("Title"):
        print(f"  Title: {c['Title']}")
    if c.get("Email"):
        print(f"  Email: {c['Email']}")

    print("\nAccount:")
    print(f"  Company: {a.get('Name') or '(none found)'}")
    if a.get("Industry"):
        print(f"  Industry: {a['Industry']}")
    if a.get("NumberOfEmployees"):
        print(f"  Team size: {a['NumberOfEmployees']} employees")
    if a.get("BudgetRange__c"):
        print(f"  Budget: {a['BudgetRange__c']}")

    print("\nOpportunity:")
    print(f"  Deal: {o.get('Name') or '(none found)'}")
    if o.get("Amount"):
        print(f"  Value: ${o['Amount']:,.0f}")
    print(f"  Stage: {o.get('StageName') or 'Prospecting'}")
    if o.get("Competitor__c"):
        print(f"  Competitor: {o['Competitor__c']}")
    if o.get("NextStep"):
        print(f"  Next step: {o['NextStep']}")
    print(f"\nExtraction confidence: {data.get('confidence', 'unknown')}")


def run_pipeline(meeting_summary: str, auto_confirm: bool = False):
    """
    Runs the full pipeline: extract -> validate -> confirm -> write.
    Set auto_confirm=True to skip the y/n prompt (useful for scripted demos).
    Returns the CRM write result dict, or None if extraction/validation
    failed or the user declined.
    """
    print("Please paste your meeting summary, and I'll extract the CRM data automatically.")
    print(f"\n> {meeting_summary}\n")

    print("Extracting...")
    try:
        extracted = extract_crm_data(meeting_summary)
    except RuntimeError as e:
        print(f"\nExtraction failed: {e}")
        return None

    try:
        warnings = validate_extraction(extracted)
    except ValidationError as e:
        print(f"\nValidation failed -- this extraction looks malformed, not writing to CRM.\nReason: {e}")
        return None

    print_extraction_summary(extracted)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if not auto_confirm:
        answer = input("\nCreate these records in the CRM? (y/n): ").strip().lower()
        if answer != "y":
            print("Cancelled -- nothing was written to the CRM.")
            return None

    crm = MockCRMClient()
    try:
        result = crm.write_crm_record(extracted)
    except RuntimeError as e:
        print(f"\n{e}")
        print("(Your extracted data above is still valid -- you can re-run the write step without re-extracting.)")
        return None

    print("\nDone. CRM actions taken:")
    for key in ("account", "contact", "opportunity"):
        record_id = result.get(f"{key}_id")
        action = result.get(f"{key}_action")
        if record_id:
            print(f"  {key.capitalize()}: {action} ({record_id})")
        else:
            print(f"  {key.capitalize()}: not created (no data found)")

    return result


if __name__ == "__main__":
    test_summary = (
        "Had a call with Sarah Johnson, Marketing Director at GrowthTech. "
        "They need marketing automation for 50-person team. Budget: $30K annually. "
        "Evaluating HubSpot vs our solution. Next: Demo scheduled for Friday."
    )
    run_pipeline(test_summary)
