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
            "Insights": {
                "type": "object",
                "description": (
                    "Sales intelligence beyond the core CRM fields. Every "
                    "string/list field here uses the literal string 'Unknown' "
                    "when the conversation doesn't state it -- never invent a "
                    "plausible-sounding value."
                ),
                "properties": {
                    "ProductOrPlan": {"type": "string"},
                    "UsersOrLicenses": {"type": "string"},
                    "Requirements": {"type": "array", "items": {"type": "string"}},
                    "InterestedFeatures": {"type": "array", "items": {"type": "string"}},
                    "PainPoints": {"type": "array", "items": {"type": "string"}},
                    "Objections": {"type": "array", "items": {"type": "string"}},
                    "CompetitorsMentioned": {"type": "array", "items": {"type": "string"}},
                    "Sentiment": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative", "Mixed", "Unknown"],
                    },
                    "BuyingIntent": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low", "Unknown"],
                    },
                    "ExpectedDecisionDate": {"type": "string"},
                    "FollowUpDate": {"type": "string"},
                },
                "required": ["ProductOrPlan", "UsersOrLicenses", "Requirements",
                              "InterestedFeatures", "PainPoints", "Objections",
                              "CompetitorsMentioned", "Sentiment", "BuyingIntent",
                              "ExpectedDecisionDate", "FollowUpDate"],
            },
        },
        "required": ["Contact", "Account", "Opportunity", "confidence", "Insights"],
    },
}

MEETING_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "participants": {
            "type": "string",
            "description": "Names, titles, and organizations of all participants in the meeting."
        },
        "purpose": {
            "type": "string",
            "description": "The primary purpose, objective, or topic of the meeting."
        },
        "customer_needs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific needs, functional requirements, and goals stated by the customer."
        },
        "pain_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Core pain points, bottlenecks, frustrations, or operational challenges discussed."
        },
        "budget": {
            "type": "string",
            "description": "Stated budget, pricing tier, cost range, or financial allocation."
        },
        "timeline": {
            "type": "string",
            "description": "Target timeline, decision deadline, rollout date, or milestone dates."
        },
        "objections": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Customer objections, concerns, hesitation, or competitor alternatives evaluated."
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key agreements, conclusions, or decisions reached during the meeting."
        },
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Agreed next action items, scheduled calls, owners, or deliverables."
        },
        "concise_summary": {
            "type": "string",
            "description": "A concise executive narrative summarizing the meeting conversation."
        },
    },
    "required": [
        "participants", "purpose", "customer_needs", "pain_points",
        "budget", "timeline", "objections", "decisions", "next_steps", "concise_summary"
    ],
}

MEETING_COACHING_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_performance_score": {
            "type": "integer",
            "description": "Rep performance score from 1-100 on discovery, qualification, and sales execution."
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific positive sales behaviors observed in this conversation."
        },
        "failed_to_ask": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific topics or critical questions the salesperson failed to ask based on what the prospect said."
        },
        "important_missing_information": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Crucial business, technical, or decision criteria info left uncovered."
        },
        "unanswered_customer_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Prospect concerns, doubts, or competitor mentions that were glossed over without deep probing."
        },
        "missed_sales_opportunities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Missed opportunities to expand deal size, multi-thread stakeholders, or accelerate deal velocity."
        },
        "recommended_questions_to_ask": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "When the customer said X or in relation to topic Y"},
                    "recommended_question": {"type": "string", "description": "Exact script of the question that should have been asked"},
                    "rationale": {"type": "string", "description": "Why this question uncovers critical deal leverage"}
                },
                "required": ["context", "recommended_question", "rationale"]
            },
            "description": "Exact scripted questions tailored directly to this conversation."
        },
        "coaching_takeaway": {
            "type": "string",
            "description": "A high-impact executive coaching summary for the salesperson."
        }
    },
    "required": [
        "overall_performance_score", "strengths", "failed_to_ask",
        "important_missing_information", "unanswered_customer_concerns",
        "missed_sales_opportunities", "recommended_questions_to_ask", "coaching_takeaway"
    ]
}