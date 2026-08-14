"""
Salesforce-style CRM extraction schema, expressed as an Anthropic tool
definition. Using a forced tool call (rather than asking the model to
"return JSON" in prose) means the model MUST return data matching this
exact shape -- no missing fields, no extra commentary mixed in.
"""

CRM_EXTRACTION_TOOL = {
    "name": "extract_crm_data",
    "description": (
        "Extract structured CRM data (Contact, Account, Opportunity) from a "
        "B2B sales meeting summary. Only include information that is "
        "explicitly stated or clearly implied in the text. Use null for "
        "any field not mentioned -- never guess or infer a value that "
        "isn't actually in the text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "Contact": {
                "type": "object",
                "properties": {
                    "FirstName": {"type": ["string", "null"]},
                    "LastName": {"type": ["string", "null"]},
                    "Title": {"type": ["string", "null"]},
                    "Department": {"type": ["string", "null"]},
                    "Email": {"type": ["string", "null"]},
                    "Phone": {"type": ["string", "null"]},
                    "PreferredContactMethod__c": {"type": ["string", "null"]},
                    "Timezone__c": {"type": ["string", "null"]},
                    "LeadSource": {"type": ["string", "null"]},
                },
                "required": ["FirstName", "LastName", "Title", "Department",
                              "Email", "Phone", "PreferredContactMethod__c",
                              "Timezone__c", "LeadSource"],
            },
            "Account": {
                "type": "object",
                "properties": {
                    "Name": {"type": ["string", "null"]},
                    "Industry": {"type": ["string", "null"]},
                    "NumberOfEmployees": {"type": ["integer", "null"]},
                    "Website": {"type": ["string", "null"]},
                    "BudgetRange__c": {"type": ["string", "null"]},
                    "PainPoints__c": {"type": ["string", "null"]},
                },
                "required": ["Name", "Industry", "NumberOfEmployees",
                              "Website", "BudgetRange__c", "PainPoints__c"],
            },
            "Opportunity": {
                "type": "object",
                "properties": {
                    "Name": {"type": ["string", "null"]},
                    "Amount": {"type": ["number", "null"]},
                    "StageName": {
                        "type": ["string", "null"],
                        "enum": ["Prospecting", "Qualification", "Needs Analysis",
                                  "Value Proposition", "Proposal/Price Quote",
                                  "Negotiation/Review", "Demo Scheduled",
                                  "Closed Won", "Closed Lost"],
                    },
                    "NextStep": {"type": ["string", "null"]},
                    "Competitor__c": {"type": ["string", "null"]},
                    "Description": {"type": ["string", "null"]},
                    "DecisionMaker__c": {
                        "type": ["object", "null"],
                        "properties": {
                            "Name": {"type": ["string", "null"]},
                            "Title": {"type": ["string", "null"]},
                        },
                    },
                },
                "required": ["Name", "Amount", "StageName", "NextStep",
                              "Competitor__c", "Description", "DecisionMaker__c"],
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Your confidence that the extraction is complete and "
                    "accurate. 'low' if the text was very sparse or ambiguous."
                ),
            },
        },
        "required": ["Contact", "Account", "Opportunity", "confidence"],
    },
}
