"""
Next Best Action (NBA) Engine for B2B Sales Conversations.
Analyzes meeting context, extracted CRM intelligence, buyer sentiment,
competitor dynamics, and deal stage to automatically determine:
1. What the salesperson should do next (Primary NBA)
2. Why (Strategic rationale & deal leverage)
3. Which structured follow-up actions are required (categorized & assigned)
"""

import json
from typing import Dict, Any, List, Optional


def determine_next_best_action(
    crm_data: Dict[str, Any],
    analysis: Dict[str, Any],
    summary: Dict[str, Any],
    coaching: Optional[Dict[str, Any]] = None,
    transcript: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes the Next Best Action (NBA) based on full meeting context,
    CRM extraction, and sales intelligence.
    """
    contact = crm_data.get("Contact", {}) or {}
    account = crm_data.get("Account", {}) or {}
    opp = crm_data.get("Opportunity", {}) or {}
    insights = crm_data.get("Insights", {}) or {}

    contact_name = f"{contact.get('FirstName') or ''} {contact.get('LastName') or ''}".strip() or "the prospect"
    first_name = contact.get("FirstName") or "the prospect"
    company_name = account.get("Name") or "the account"
    stage = opp.get("StageName") or "Prospecting"
    next_step = opp.get("NextStep") or (summary.get("next_steps", ["Follow-up with customer"])[0] if summary else "Follow-up with customer")
    competitor = opp.get("Competitor__c") or (analysis.get("competitors_mentioned", [None])[0] if analysis.get("competitors_mentioned") else None)
    
    amount = opp.get("Amount")
    budget_str = f"${amount:,.0f}" if isinstance(amount, (int, float)) else (account.get("BudgetRange__c") or "stated budget")
    sentiment = analysis.get("sentiment", "Positive")
    buying_intent = analysis.get("buying_intent", "High")

    # 1. Determine Primary Next Best Action & Strategic Rationale
    if competitor and "demo" in next_step.lower():
        primary_action = f"Confirm Friday Demo & Prepare Tailored {competitor} Differentiation Showcase"
        rationale = (
            f"{first_name} at {company_name} has verified budget approval ({budget_str}) and indicated dissatisfaction with {competitor}'s rigid reporting. "
            f"Delivering a tailored ops walkthrough addressing manual workflow friction while showcasing reporting advantages directly neutralizes {competitor}."
        )
        urgency = "High (Immediate Next 24 Hours)"
    elif "demo" in next_step.lower() or "walkthrough" in next_step.lower():
        primary_action = f"Execute Technical & Ops Walkthrough for {company_name}"
        rationale = (
            f"Buying intent is {buying_intent} with verified budget of {budget_str}. Demonstrating exact solution fit for their team will transition this deal from qualification to proposal."
        )
        urgency = "High (Schedule within 48 Hours)"
    elif "proposal" in stage.lower() or "quote" in stage.lower():
        primary_action = f"Deliver Formal Commercial Proposal to {company_name}"
        rationale = f"Requirements and commercial allocation of {budget_str} are aligned. Delivering a structured quote locks in budget before end of cycle."
        urgency = "Medium (Next 3 Days)"
    else:
        primary_action = f"Send Meeting Brief & Lock In Next Milestone: {next_step}"
        rationale = f"Maintains momentum with {contact_name} by confirming agreed deliverables and establishing decision timeline."
        urgency = "Standard (Within 24 Hours)"

    # 2. Categorized Required Follow-Up Actions
    required_follow_ups = []

    # Category A: Immediate Customer Outreach
    required_follow_ups.append({
        "category": "Immediate Customer Outreach",
        "action": f"Send personalized follow-up email to {first_name} confirming: '{next_step}'.",
        "suggested_owner": "Account Executive",
        "deadline": "Within 2 Hours",
        "status": "Ready to Send"
    })

    # Category B: Competitive Differentiation / Prep
    if competitor:
        required_follow_ups.append({
            "category": "Competitive Differentiation",
            "action": f"Prepare side-by-side battlecard and reporting feature matrix highlighting key advantages over {competitor}.",
            "suggested_owner": "Sales Engineer / AE",
            "deadline": "Prior to Next Scheduled Demo",
            "status": "Action Required"
        })

    # Category C: Stakeholder Multi-Threading
    if "vp" in str(summary).lower() or "vp" in str(crm_data).lower():
        required_follow_ups.append({
            "category": "Stakeholder Multi-Threading",
            "action": f"Invite the VP who approved the {budget_str} budget to the executive summary portion of the upcoming walkthrough.",
            "suggested_owner": "Account Executive",
            "deadline": "During Demo Confirmation",
            "status": "Recommended"
        })
    else:
        required_follow_ups.append({
            "category": "Stakeholder Discovery",
            "action": f"Identify economic buyer and procurement decision maker at {company_name}.",
            "suggested_owner": "Sales Rep",
            "deadline": "Next Touchpoint",
            "status": "Discovery"
        })

    # Category D: CRM & Pipeline Hygiene
    required_follow_ups.append({
        "category": "CRM & Pipeline Automation",
        "action": f"Upsert Account '{company_name}', Contact '{contact_name}', and set Opportunity stage to '{stage}' with deal value {budget_str}.",
        "suggested_owner": "Automated Agent",
        "deadline": "Immediate",
        "status": "Ready for Execution"
    })

    # 3. Deal Acceleration Strategy
    deal_acceleration_strategy = (
        f"Prioritize live demonstration of automated workflow triggers and executive reporting in the first 10 minutes of the next conversation. "
        f"This directly solves {company_name}'s stated pain point of manual onboarding overhead."
    )

    return {
        "primary_action": primary_action,
        "rationale": rationale,
        "urgency": urgency,
        "target_milestone": next_step,
        "required_follow_ups": required_follow_ups,
        "deal_acceleration_strategy": deal_acceleration_strategy,
    }
