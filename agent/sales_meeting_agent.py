"""
AI Sales Meeting Agent
======================
Converts raw B2B sales conversations and meeting transcripts into actionable
intelligence by orchestrating core agentic steps:

1. Automatic Meeting Summarization: Automatically extracts participants, purpose,
   customer needs, pain points, budget, timeline, objections, decisions, and next steps.
2. Conversation Analysis: Analyzes meeting type, sentiment, stakeholder engagement, and urgency.
3. Information Need Assessment: Evaluates what CRM fields & sales intelligence are complete vs missing.
4. CRM Data Extraction: Reuses Llama 3.2 Ollama extractor & schema validation.
5. Action & Decision Engine: Recommends CRM actions, sales tasks, risk mitigations, and context-aware follow-up email drafts.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from agent.extractor_ollama import extract_crm_data as extract_crm_via_ollama
from agent.validator import validate_extraction, ValidationError
from agent.mock_crm_client import MockCRMClient
from agent.summarizer import summarize_meeting_transcript
from agent.coaching import generate_meeting_coaching
from agent.next_best_action import determine_next_best_action
from agent.transcriber import transcribe_meeting_audio


@dataclass
class AgentResult:
    """Encapsulates the complete result of the AI Sales Meeting Agent pipeline."""
    raw_text: str
    analysis: Dict[str, Any]
    information_needed: Dict[str, Any]
    crm_data: Dict[str, Any]
    validation_warnings: List[str]
    summary: Dict[str, Any]
    actions: Dict[str, Any]
    coaching: Optional[Dict[str, Any]] = None
    crm_write_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "analysis": self.analysis,
            "information_needed": self.information_needed,
            "crm_data": self.crm_data,
            "validation_warnings": self.validation_warnings,
            "summary": self.summary,
            "actions": self.actions,
            "coaching": self.coaching,
            "crm_write_result": self.crm_write_result,
        }


class SalesMeetingAgent:
    """
    Intelligent agent that processes B2B sales meeting transcripts,
    automatically summarizes conversations, assesses completeness,
    extracts CRM data with Llama 3.2, and decides required sales actions.
    """

    def __init__(self, model_name: str = "llama3.2", crm_client: Optional[MockCRMClient] = None):
        self.model_name = model_name
        self.crm_client = crm_client or MockCRMClient()

    # -------------------------------------------------------------------------
    # Automatic Meeting Summarization
    # -------------------------------------------------------------------------
    def summarize_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Automatically generates a structured, comprehensive summary from raw
        meeting transcripts or conversations covering:
        - participants
        - purpose
        - customer_needs
        - pain_points
        - budget
        - timeline
        - objections
        - decisions
        - next_steps
        - concise_summary
        """
        return summarize_meeting_transcript(transcript, model=self.model_name)

    # -------------------------------------------------------------------------
    # Step 1: Conversation Analysis
    # -------------------------------------------------------------------------
    def analyze_conversation(
        self,
        text: str,
        extracted_data: Optional[Dict[str, Any]] = None,
        auto_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes conversation context, meeting type, sentiment, stakeholder roles,
        and buying signals.
        """
        lower = text.lower()

        # Determine Meeting Type from auto_summary or keywords
        purpose = auto_summary.get("purpose") if auto_summary else None
        if purpose:
            meeting_type = purpose
        elif any(w in lower for w in ["demo", "walkthrough", "product tour", "showed", "presented"]):
            meeting_type = "Product Demo / Presentation"
        elif any(w in lower for w in ["pricing", "proposal", "contract", "quote", "discount", "budget", "procurement"]):
            meeting_type = "Pricing / Proposal Negotiation"
        elif any(w in lower for w in ["support", "issue", "ticket", "bug", "onboarding"]):
            meeting_type = "Customer Success / Onboarding"
        elif any(w in lower for w in ["intro", "discovery", "initial call", "exploratory", "evaluating"]):
            meeting_type = "Discovery & Qualification"
        else:
            meeting_type = "General Sales Sync"

        # Determine Urgency & Timeline
        timeline_str = (auto_summary.get("timeline") if auto_summary else "").lower()
        if any(w in lower or w in timeline_str for w in ["asap", "urgent", "this week", "immediately", "friday", "tomorrow"]):
            urgency = "High (Immediate Next Steps)"
        elif any(w in lower or w in timeline_str for w in ["next month", "q1", "q2", "q3", "q4", "end of quarter", "few weeks"]):
            urgency = "Medium (Targeted Cycle)"
        else:
            urgency = "Normal / Standard Cadence"

        # Extract Insights from extracted data if present
        insights = (extracted_data or {}).get("Insights", {})
        sentiment = insights.get("Sentiment")
        if not sentiment or sentiment == "Unknown":
            if any(w in lower for w in ["great", "excited", "loved", "impressed", "better", "fit", "strong"]):
                sentiment = "Positive"
            elif any(w in lower for w in ["concerned", "expensive", "skeptical", "delay", "blocked", "issue"]):
                sentiment = "Mixed / Hesitant"
            elif any(w in lower for w in ["no", "cancelled", "declined", "pass", "dropped"]):
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

        buying_intent = insights.get("BuyingIntent")
        if not buying_intent or buying_intent == "Unknown":
            if any(w in lower for w in ["ready to buy", "approved budget", "budget approved", "move forward", "send contract"]):
                buying_intent = "High"
            elif any(w in lower for w in ["evaluating", "comparing", "demo scheduled", "interested", "looking for"]):
                buying_intent = "Medium"
            else:
                buying_intent = "Low / Exploratory"

        # Competitor Presence
        competitors_mentioned = list(insights.get("CompetitorsMentioned") or [])
        if auto_summary and auto_summary.get("objections"):
            for obj in auto_summary["objections"]:
                for comp in ["HubSpot", "Salesforce", "Marketo", "Pipedrive", "Zendesk", "Zoho", "ActiveCampaign"]:
                    if comp.lower() in str(obj).lower() and comp not in competitors_mentioned:
                        competitors_mentioned.append(comp)

        if not competitors_mentioned:
            for comp in ["HubSpot", "Salesforce", "Marketo", "Pipedrive", "Zendesk", "Zoho", "ActiveCampaign"]:
                if comp.lower() in lower:
                    competitors_mentioned.append(comp)

        return {
            "meeting_type": meeting_type,
            "sentiment": sentiment,
            "buying_intent": buying_intent,
            "urgency": urgency,
            "competitors_mentioned": competitors_mentioned,
            "deal_signals": [
                f"Meeting classified as: {meeting_type}",
                f"Buyer tone assessed as: {sentiment}",
                f"Intent level: {buying_intent}",
                f"Engagement urgency: {urgency}",
            ]
        }

    # -------------------------------------------------------------------------
    # Step 2: Information Need Assessment
    # -------------------------------------------------------------------------
    def decide_needed_information(
        self,
        text: str,
        crm_data: Dict[str, Any],
        analysis: Dict[str, Any],
        auto_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates what information is needed for complete CRM records and deals,
        identifying captured data vs missing critical gaps.
        """
        contact = crm_data.get("Contact", {}) or {}
        account = crm_data.get("Account", {}) or {}
        opp = crm_data.get("Opportunity", {}) or {}
        insights = crm_data.get("Insights", {}) or {}

        captured_fields = []
        missing_critical_fields = []
        follow_up_discovery_questions = []

        # Contact checks
        if contact.get("FirstName") or contact.get("LastName"):
            captured_fields.append("Contact Full Name")
        elif auto_summary and auto_summary.get("participants") and auto_summary["participants"] != "Not specified":
            captured_fields.append(f"Participants: {auto_summary['participants']}")
        else:
            missing_critical_fields.append("Contact Name (Prospect identity unknown)")
            follow_up_discovery_questions.append("Who is the primary contact person for this opportunity?")

        if contact.get("Title"):
            captured_fields.append(f"Contact Title: {contact['Title']}")
        else:
            missing_critical_fields.append("Contact Title & Seniority")
            follow_up_discovery_questions.append("What is the prospect's exact job title and decision authority?")

        if contact.get("Email"):
            captured_fields.append(f"Contact Email: {contact['Email']}")
        else:
            missing_critical_fields.append("Direct Email Address")
            follow_up_discovery_questions.append("What is the direct email address to send materials/calendar invites?")

        # Account checks
        if account.get("Name"):
            captured_fields.append(f"Company Name: {account['Name']}")
        else:
            missing_critical_fields.append("Company Name")
            follow_up_discovery_questions.append("What company or organization are they representing?")

        if account.get("NumberOfEmployees"):
            captured_fields.append(f"Team / Company Size: {account['NumberOfEmployees']} employees")
        else:
            missing_critical_fields.append("Team Size / Number of Employees")

        if account.get("BudgetRange__c") or opp.get("Amount"):
            amt = opp.get("Amount")
            val_str = f"${amt:,.0f}" if amt is not None else account.get("BudgetRange__c")
            captured_fields.append(f"Budget / Deal Value: {val_str}")
        elif auto_summary and auto_summary.get("budget") and "not" not in auto_summary["budget"].lower():
            captured_fields.append(f"Budget: {auto_summary['budget']}")
        else:
            missing_critical_fields.append("Budget / Financial Allocation")
            follow_up_discovery_questions.append("Has a specific budget range or spending limit been earmarked for this solution?")

        # Opportunity checks
        if opp.get("StageName"):
            captured_fields.append(f"Sales Stage: {opp['StageName']}")
        else:
            missing_critical_fields.append("Definitive Deal Stage")

        if opp.get("NextStep"):
            captured_fields.append(f"Explicit Next Step: {opp['NextStep']}")
        elif auto_summary and auto_summary.get("next_steps") and auto_summary["next_steps"]:
            captured_fields.append(f"Next Steps: {', '.join(auto_summary['next_steps'])}")
        else:
            missing_critical_fields.append("Clear Next Step & Owner")
            follow_up_discovery_questions.append("What is the specific next milestone and scheduled date?")

        # Decision Maker check
        decision_maker = opp.get("DecisionMaker__c")
        if decision_maker and isinstance(decision_maker, dict) and decision_maker.get("Name"):
            captured_fields.append(f"Decision Maker: {decision_maker['Name']}")
        elif "vp" in text.lower() or "exec" in text.lower() or "director" in text.lower():
            captured_fields.append("Decision Maker authority mentioned in text")
        else:
            missing_critical_fields.append("Economic Buyer / Decision Maker verification")
            follow_up_discovery_questions.append("Who is the final economic decision maker approving the purchase?")

        # Completeness Score Calculation
        total_checks = len(captured_fields) + len(missing_critical_fields)
        completeness_percentage = int((len(captured_fields) / total_checks * 100)) if total_checks > 0 else 50

        status = "Complete" if completeness_percentage >= 80 else ("Partially Complete" if completeness_percentage >= 50 else "Needs Major Enrichment")

        return {
            "completeness_score": completeness_percentage,
            "status": status,
            "captured_fields": captured_fields,
            "missing_critical_fields": missing_critical_fields,
            "recommended_discovery_questions": follow_up_discovery_questions,
            "summary_assessment": f"{len(captured_fields)} key attributes captured ({completeness_percentage}% complete); {len(missing_critical_fields)} missing items require sales attention.",
        }

    # -------------------------------------------------------------------------
    # Step 3: CRM Data Extraction
    # -------------------------------------------------------------------------
    def extract_crm_data(self, text: str) -> Dict[str, Any]:
        """
        Calls the Llama 3.2 Ollama extractor and returns validated structured CRM data.
        """
        extracted = extract_crm_via_ollama(text, model=self.model_name)
        return extracted

    # -------------------------------------------------------------------------
    # Step 4: Executive Meeting Summary Generation
    # -------------------------------------------------------------------------
    def generate_summary(
        self,
        text: str,
        crm_data: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
        auto_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a concise executive meeting summary covering participants,
        purpose, customer needs, pain points, budget, timeline, objections,
        decisions, and next steps automatically.
        """
        contact = crm_data.get("Contact", {}) or {}
        account = crm_data.get("Account", {}) or {}
        opp = crm_data.get("Opportunity", {}) or {}
        insights = crm_data.get("Insights", {}) or {}

        contact_name = f"{contact.get('FirstName') or ''} {contact.get('LastName') or ''}".strip() or "Prospect"
        company_name = account.get("Name") or "Client Organization"
        title = contact.get("Title") or "Stakeholder"
        deal_amount = opp.get("Amount")
        amount_str = f"${deal_amount:,.0f}" if isinstance(deal_amount, (int, float)) else (account.get("BudgetRange__c") or "Unspecified")

        # Automatically generated summary fields from Llama 3.2
        sum_data = auto_summary or self.summarize_transcript(text)

        participants = sum_data.get("participants") or f"{contact_name} ({title}) @ {company_name}"
        purpose = sum_data.get("purpose") or "Sales Qualification & Solution Evaluation"
        customer_needs = list(sum_data.get("customer_needs") or insights.get("Requirements") or ["Evaluate automated solution to streamline workflows"])
        pain_points = list(sum_data.get("pain_points") or insights.get("PainPoints") or ["Manual workflows causing team friction"])
        budget_summary = sum_data.get("budget") or amount_str
        timeline_summary = sum_data.get("timeline") or insights.get("FollowUpDate") or opp.get("NextStep") or "Upcoming milestone"
        objections = list(sum_data.get("objections") or insights.get("Objections") or [])
        decisions = list(sum_data.get("decisions") or [])
        next_steps = list(sum_data.get("next_steps") or ([opp.get("NextStep")] if opp.get("NextStep") else ["Follow-up with customer"]))

        concise = sum_data.get("concise_summary")
        if not concise:
            concise = (
                f"Met with {contact_name} ({title}) from {company_name} regarding {purpose}. "
                f"The team is actively exploring solutions with an estimated budget of {amount_str}. "
                f"Current deal stage is '{opp.get('StageName') or 'Prospecting'}' with next step: '{opp.get('NextStep') or 'Follow-up scheduled'}'."
            )

        competitors = analysis.get("competitors_mentioned", []) if analysis else (insights.get("CompetitorsMentioned") or [])

        return {
            "executive_overview": concise,
            "concise_summary": concise,
            "participants": participants,
            "key_participants": participants,
            "purpose": purpose,
            "customer_needs": customer_needs,
            "solution_alignment": customer_needs,
            "pain_points": pain_points,
            "core_pain_points": pain_points,
            "budget": budget_summary,
            "timeline": timeline_summary,
            "objections": objections,
            "decisions": decisions,
            "next_steps": next_steps,
            "budget_and_commercials": {
                "budget": budget_summary,
                "stage": opp.get("StageName") or "Prospecting",
                "target_next_step": next_steps[0] if next_steps else "Follow-up",
            },
            "competitive_notes": f"Competing with {', '.join(competitors)}" if competitors else "No active competitors cited.",
        }

    # -------------------------------------------------------------------------
    # Step 5: Action & Decision Engine
    # -------------------------------------------------------------------------
    def decide_actions(
        self,
        crm_data: Dict[str, Any],
        analysis: Dict[str, Any],
        info_assessment: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decides what specific CRM, sales, follow-up, and risk mitigation actions
        must be taken following this meeting.
        """
        contact = crm_data.get("Contact", {}) or {}
        account = crm_data.get("Account", {}) or {}
        opp = crm_data.get("Opportunity", {}) or {}
        insights = crm_data.get("Insights", {}) or {}

        first_name = contact.get("FirstName") or "there"
        company = account.get("Name") or "your company"
        next_step = opp.get("NextStep") or (summary.get("next_steps", ["our follow-up conversation"])[0] if summary else "our follow-up conversation")
        competitor = opp.get("Competitor__c") or (analysis.get("competitors_mentioned", [None])[0] if analysis.get("competitors_mentioned") else None)
        deal_amount = opp.get("Amount")
        budget_str = f"${deal_amount:,.0f}" if isinstance(deal_amount, (int, float)) else account.get("BudgetRange__c")

        # 1. CRM Actions
        crm_actions = []
        if account.get("Name"):
            crm_actions.append(f"Upsert Account record: '{account['Name']}'")
        if contact.get("FirstName") or contact.get("LastName"):
            contact_full_name = f"{first_name} {contact.get('LastName') or ''}".strip()
            crm_actions.append(f"Upsert Contact record: '{contact_full_name}'")
        if opp.get("Name") or opp.get("StageName"):
            crm_actions.append(f"Create Opportunity '{opp.get('Name') or 'Sales Opportunity'}' at stage '{opp.get('StageName') or 'Prospecting'}'")

        # 2. Recommended Sales Tasks & Next Steps
        tasks = []
        if next_step:
            tasks.append({
                "title": f"Execute Next Step: {next_step}",
                "priority": "High",
                "suggested_owner": "Account Executive",
                "due_timeline": "Immediate / Scheduled milestone",
            })

        if competitor:
            tasks.append({
                "title": f"Prepare Competitive Battlecard vs {competitor}",
                "priority": "Medium",
                "suggested_owner": "Sales Engineer / AE",
                "due_timeline": "Prior to next scheduled call",
            })

        if info_assessment.get("missing_critical_fields"):
            tasks.append({
                "title": f"Enrich Missing CRM Intelligence ({len(info_assessment['missing_critical_fields'])} fields)",
                "priority": "Medium",
                "suggested_owner": "Sales Rep",
                "due_timeline": "Next Discovery Touchpoint",
            })

        # 3. Dynamic Tailored Follow-Up Email
        comp_paragraph = (
            f"\nI understand you're currently evaluating {competitor}. I'd be glad to share a quick side-by-side comparison illustrating why teams choose us over {competitor}."
            if competitor else ""
        )
        budget_mention = f"with your target budget of {budget_str}" if budget_str else "for your team's goals"

        email_subject = f"Next steps for {company} | Great connecting today"
        email_body = f"""Hi {first_name},

Thank you for taking the time to speak today. It was great learning more about {company}'s current initiatives.

Based on our discussion, our solution is well positioned to deliver value {budget_mention}.{comp_paragraph}

As agreed, our next step is: {next_step}.

Looking forward to continuing our collaboration. Please let me know if you need any additional materials in the meantime.

Best regards,
Your Dedicated Sales Team"""

        # 4. Strategic Risk Alerts
        risk_alerts = []
        if competitor:
            risk_alerts.append(f"Competitive pressure: Active evaluation of {competitor} identified. Position key differentiators early.")
        if analysis.get("buying_intent") in ["Low", "Low / Exploratory"]:
            risk_alerts.append("Low immediate buying intent detected: prioritize value demonstration and stakeholder alignment.")
        if not contact.get("Email"):
            risk_alerts.append("Missing prospect email: ensure direct contact channel is secured.")

        # 5. Next Best Action (NBA) Engine
        nba = determine_next_best_action(
            crm_data=crm_data,
            analysis=analysis,
            summary=summary or {}
        )

        return {
            "primary_next_best_action": nba.get("primary_action"),
            "next_best_action": nba,
            "crm_actions": crm_actions,
            "tasks": tasks,
            "follow_up_email": {
                "subject": email_subject,
                "body": email_body,
            },
            "risk_alerts": risk_alerts,
            "action_summary": f"{len(crm_actions)} CRM record actions, {len(tasks)} immediate sales tasks, and Next Best Action determined.",
        }

    # -------------------------------------------------------------------------
    # Step 6: Meeting Coaching & Sales Performance Feedback
    # -------------------------------------------------------------------------
    def generate_coaching(
        self,
        transcript: str,
        crm_data: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the conversation from a sales coach perspective:
        identifies what the rep failed to ask, missing information,
        unanswered concerns, missed sales opportunities, and exact
        recommended questions to ask.
        """
        return generate_meeting_coaching(
            transcript,
            crm_data=crm_data,
            summary=summary,
            model=self.model_name
        )

    # -------------------------------------------------------------------------
    # Orchestrated Agent Execution
    # -------------------------------------------------------------------------
    def process_meeting(
        self,
        transcript_or_summary: str,
        auto_confirm: bool = True,
        crm_client: Optional[MockCRMClient] = None
    ) -> AgentResult:
        """
        Orchestrates the entire AI Sales Meeting Agent pipeline:
        1. Automatic Meeting Summarization (from raw transcript or notes)
        2. CRM Data Extraction & Validation (reusing Llama 3.2)
        3. Conversation Analysis
        4. Information Need Assessment
        5. Executive Meeting Summary Generation
        6. Action & Decision Determination
        7. Meeting Coaching & Sales Performance Feedback
        8. Automated CRM Execution: Determines create vs update for Contact,
           Account, Opportunity, and Action Items.
        """
        text = transcript_or_summary.strip()
        client = crm_client or self.crm_client

        # 1. Automatic Meeting Summarization from transcript
        auto_summary = self.summarize_transcript(text)

        # 2. CRM Extraction (reusing Llama 3.2 engine)
        crm_data = self.extract_crm_data(text)

        # 3. Validation
        validation_warnings = validate_extraction(crm_data)

        # 4. Conversation Analysis
        analysis = self.analyze_conversation(text, extracted_data=crm_data, auto_summary=auto_summary)

        # 5. Information Need Assessment
        info_assessment = self.decide_needed_information(text, crm_data=crm_data, analysis=analysis, auto_summary=auto_summary)

        # 6. Executive Summary Generation (enriched with all 9 key summary items)
        summary = self.generate_summary(text, crm_data=crm_data, analysis=analysis, auto_summary=auto_summary)

        # 7. Action Decisions & Next Best Action
        actions = self.decide_actions(crm_data=crm_data, analysis=analysis, info_assessment=info_assessment, summary=summary)

        # 8. Meeting Coaching & Sales Performance Analysis
        coaching = self.generate_coaching(text, crm_data=crm_data, summary=summary)

        # 9. Automated CRM Operations (Create / Update for Contact, Account, Opportunity, Action Items)
        crm_result = None
        if auto_confirm:
            try:
                crm_result = client.write_crm_record(crm_data, actions=actions)
            except Exception as e:
                validation_warnings.append(f"Auto-write to CRM failed: {e}")

        return AgentResult(
            raw_text=text,
            analysis=analysis,
            information_needed=info_assessment,
            crm_data=crm_data,
            validation_warnings=validation_warnings,
            summary=summary,
            actions=actions,
            coaching=coaching,
            crm_write_result=crm_result,
        )

    def process_meeting_audio(
        self,
        audio_source: Any,
        model_size: str = "base",
        auto_confirm: bool = True,
        crm_client: Optional[MockCRMClient] = None
    ) -> AgentResult:
        """
        Transcribes a meeting audio recording using Whisper and automatically
        feeds the resulting transcript into the complete Sales Meeting Agent pipeline.
        """
        transcribe_result = transcribe_meeting_audio(audio_source, model_size=model_size)
        if not transcribe_result.get("success"):
            err = transcribe_result.get("error", "Unknown transcription error")
            raise RuntimeError(f"Speech-to-text transcription failed: {err}")

        transcript_text = transcribe_result.get("transcript", "")
        if not transcript_text.strip():
            raise ValueError("Whisper transcribed empty audio. Please provide a clear audio recording.")

        return self.process_meeting(
            transcript_text,
            auto_confirm=auto_confirm,
            crm_client=crm_client
        )

    def run_end_to_end_pipeline(
        self,
        meeting_input: Any,
        auto_confirm: bool = True,
        crm_client: Optional[MockCRMClient] = None
    ) -> AgentResult:
        """
        Unified automatic pipeline entry point:
        Meeting Audio / Recording -> Speech-to-Text -> Meeting Summary ->
        AI Sales Agent -> CRM Extraction -> Missed Questions ->
        What We Should Have Said -> Next Best Action -> Action Items.
        
        The user provides only the meeting recording or conversation notes.
        """
        is_audio = False
        if isinstance(meeting_input, (bytes, bytearray)):
            is_audio = True
        elif hasattr(meeting_input, "read"):
            is_audio = True
        elif isinstance(meeting_input, (str, Path)):
            ext = str(meeting_input).lower()
            if any(ext.endswith(sfx) for sfx in [".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"]):
                is_audio = True

        if is_audio:
            return self.process_meeting_audio(
                meeting_input,
                auto_confirm=auto_confirm,
                crm_client=crm_client
            )
        else:
            return self.process_meeting(
                str(meeting_input),
                auto_confirm=auto_confirm,
                crm_client=crm_client
            )



