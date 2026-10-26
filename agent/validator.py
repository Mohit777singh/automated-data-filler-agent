"""
Validates output from extract_crm_data() before it's allowed to proceed
to CRM writing. Catches malformed data early rather than letting bad
records reach Salesforce.
"""

VALID_STAGES = {
    "Prospecting", "Qualification", "Needs Analysis", "Value Proposition",
    "Proposal/Price Quote", "Negotiation/Review", "Demo Scheduled",
    "Closed Won", "Closed Lost", None,
}

REQUIRED_TOP_KEYS = {"Contact", "Account", "Opportunity", "confidence"}


class ValidationError(Exception):
    pass


def validate_extraction(data: dict) -> list:
    """
    Returns a list of warning strings (empty list = fully clean).
    Raises ValidationError for structural problems that make the
    record unsafe to write to a CRM at all.
    """
    warnings = []

    missing_keys = REQUIRED_TOP_KEYS - set(data.keys())
    if missing_keys:
        raise ValidationError(f"Missing top-level keys: {missing_keys}")

    contact, account, opp = data["Contact"], data["Account"], data["Opportunity"]

    amount = opp.get("Amount")
    if amount is not None and not isinstance(amount, (int, float)):
        raise ValidationError(f"Opportunity.Amount must be a number or null, got: {amount!r}")

    employees = account.get("NumberOfEmployees")
    if employees is not None and not isinstance(employees, int):
        raise ValidationError(f"Account.NumberOfEmployees must be an int or null, got: {employees!r}")

    stage = opp.get("StageName")
    if stage not in VALID_STAGES:
        raise ValidationError(f"Invalid StageName: {stage!r}")

    if data.get("confidence") == "low":
        warnings.append("Low confidence extraction -- recommend human review before writing to CRM.")

    has_contact_name = contact.get("FirstName") or contact.get("LastName")
    has_company = account.get("Name")
    if not has_contact_name and not has_company:
        warnings.append("No contact name or company name extracted -- likely not a real sales lead.")

    email = contact.get("Email")
    if email and "@" not in email:
        warnings.append(f"Contact.Email looks malformed: {email!r}")

    if opp.get("Amount") is not None and opp.get("Amount") < 0:
        raise ValidationError(f"Opportunity.Amount cannot be negative: {opp.get('Amount')}")

    # Insights is additive sales intelligence -- never blocks a CRM write.
    # Missing/malformed Insights just means the UI won't have extra detail
    # to show; the core Contact/Account/Opportunity write can still proceed.
    insights = data.get("Insights")
    if insights is not None:
        valid_sentiments = {"Positive", "Neutral", "Negative", "Mixed", "Unknown"}
        if insights.get("Sentiment") and insights["Sentiment"] not in valid_sentiments:
            warnings.append(f"Insights.Sentiment has an unexpected value: {insights.get('Sentiment')!r}")

        valid_intents = {"High", "Medium", "Low", "Unknown"}
        if insights.get("BuyingIntent") and insights["BuyingIntent"] not in valid_intents:
            warnings.append(f"Insights.BuyingIntent has an unexpected value: {insights.get('BuyingIntent')!r}")

        for list_field in ("Requirements", "InterestedFeatures", "PainPoints",
                            "Objections", "CompetitorsMentioned"):
            val = insights.get(list_field)
            if val is not None and not isinstance(val, list):
                warnings.append(f"Insights.{list_field} should be a list, got: {type(val).__name__}")
    else:
        warnings.append("No Insights data returned -- AI Insights panel will be empty for this extraction.")

    return warnings


if __name__ == "__main__":
    good = {
        "Contact": {"FirstName": "Sarah", "LastName": "Johnson", "Title": "Marketing Director",
                     "Department": "Marketing", "Email": "sarah@growthtech.com", "Phone": None,
                     "PreferredContactMethod__c": None, "Timezone__c": None, "LeadSource": None},
        "Account": {"Name": "GrowthTech", "Industry": None, "NumberOfEmployees": 50,
                     "Website": None, "BudgetRange__c": "$30K annually", "PainPoints__c": None},
        "Opportunity": {"Name": "Marketing Automation - GrowthTech", "Amount": 30000,
                          "StageName": "Demo Scheduled", "NextStep": "Demo on Friday",
                          "Competitor__c": "HubSpot", "Description": None, "DecisionMaker__c": None},
        "confidence": "high",
    }
    print("Warnings:", validate_extraction(good))