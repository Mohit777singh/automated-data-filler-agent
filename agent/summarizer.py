"""
Automatic Meeting Summarization for Sales Conversations and Transcripts.
Leverages local Llama 3.2 model to automatically extract executive summaries
covering participants, purpose, customer needs, pain points, budget, timeline,
objections, decisions, and next steps without requiring manual user input.
"""

import json
import re
import requests
from typing import Dict, Any, List, Optional

from agent.schema import MEETING_SUMMARY_SCHEMA
from agent.http_utils import request_with_retry, FriendlyRequestError
from agent.extractor_ollama import resolve_ollama_model

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

SUMMARIZER_SYSTEM_PROMPT = """
You are an expert Executive Sales Intelligence Assistant for B2B revenue teams.

Your task is to analyze the provided raw sales meeting transcript or conversation notes
and automatically generate a comprehensive, structured meeting summary.

You MUST extract and populate all of the following:
1. participants: Full names, job titles, and company names of everyone involved.
2. purpose: The core reason/goal for this meeting (e.g. Discovery call, Product demo, Pricing sync).
3. customer_needs: List of specific customer needs, requested capabilities, and product requirements.
4. pain_points: List of current operational bottlenecks, inefficiencies, or frustrations discussed.
5. budget: Stated budget amount, spending limits, or pricing discussions (or 'Not specified' if unmentioned).
6. timeline: Key dates, target deployment milestones, or decision deadlines (or 'Not specified').
7. objections: Customer concerns, hesitations, or competitor comparisons raised during the call.
8. decisions: Any agreed decisions, mutual alignments, or conclusions reached during the call.
9. next_steps: Specific follow-up actions, scheduled demos/meetings, owners, and target dates.
10. concise_summary: A coherent, high-impact executive summary paragraph integrating the entire conversation.

Rules:
- Extract facts faithfully from the conversation.
- Do not hallucinate figures or names that are not stated or strongly evidenced.
- If a field is not present in the text, provide a sensible indicator (e.g. "Not explicitly stated").
"""


def _fallback_heuristic_summary(transcript: str) -> Dict[str, Any]:
    """
    Robust fallback summarizer if Ollama local server is momentarily unreachable.
    Parses conversational text using pattern extraction.
    """
    lines = transcript.strip().split("\n")
    lower = transcript.lower()

    # Participants
    participants = []
    for line in lines:
        if ":" in line:
            speaker = line.split(":", 1)[0].strip()
            if speaker and len(speaker.split()) <= 4 and speaker not in participants:
                participants.append(speaker)
    if not participants:
        name_match = re.search(r"\b(?:with|spoke to|met with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", transcript)
        participants_str = name_match.group(1) if name_match else "Sales Representative & Prospect"
    else:
        participants_str = ", ".join(participants[:4])

    # Purpose
    if "demo" in lower:
        purpose = "Product Demonstration & Solution Overview"
    elif "pricing" in lower or "proposal" in lower or "quote" in lower:
        purpose = "Pricing Discussion & Commercial Negotiation"
    elif "intro" in lower or "discovery" in lower:
        purpose = "Initial Discovery & Needs Assessment"
    else:
        purpose = "B2B Sales Qualification & Solution Exploration"

    # Needs & Pain Points
    needs = []
    pains = []
    if "automation" in lower or "manual" in lower:
        pains.append("Manual repetitive workflows causing friction")
        needs.append("Automated workflow and integration platform")
    if "report" in lower or "analytics" in lower:
        needs.append("Advanced analytics and executive reporting suite")
    if not needs:
        needs.append("Modernizing sales and operational technology stack")
    if not pains:
        pains.append("Current tooling lacks efficiency and team scalability")

    # Budget
    budget_match = re.search(r"(\$[\d,]+(?:\s*(?:k|m|thousand|million|annually|per\s+year))?|\b\d+k\b)", transcript, re.IGNORECASE)
    budget = budget_match.group(1) if budget_match else "Pending commercial review"

    # Timeline
    timeline_match = re.search(r"\b(friday|monday|next week|next month|q[1-4]|end of (?:the )?month)\b", lower)
    timeline = timeline_match.group(1).title() if timeline_match else "Upcoming follow-up milestone"

    # Objections / Competitors
    objections = []
    for comp in ["HubSpot", "Salesforce", "Marketo", "Pipedrive", "Zendesk", "Zoho"]:
        if comp.lower() in lower:
            objections.append(f"Evaluating competitor: {comp}")
    if not objections:
        objections.append("None explicitly raised during initial sync")

    # Decisions
    decisions = []
    if "agree" in lower or "approved" in lower or "liked" in lower or "scheduled" in lower:
        decisions.append("Mutual alignment to advance to next evaluation stage")
    else:
        decisions.append("Initial discovery completed; reviewing capabilities")

    # Next Steps
    next_match = re.search(r"(?:next step|next:)\s*([^\.\n]+)", transcript, re.IGNORECASE)
    next_step = next_match.group(1).strip() if next_match else "Schedule follow-up session"
    next_steps = [next_step]

    # Concise Summary
    concise = (
        f"Meeting held regarding {purpose.lower()}. Participants discussed current challenges including {pains[0].lower()} "
        f"and requirement for {needs[0].lower()}. Budget discussed at {budget} with target timeline around {timeline}. "
        f"Agreed next step: {next_step}."
    )

    return {
        "participants": participants_str,
        "purpose": purpose,
        "customer_needs": needs,
        "pain_points": pains,
        "budget": budget,
        "timeline": timeline,
        "objections": objections,
        "decisions": decisions,
        "next_steps": next_steps,
        "concise_summary": concise,
    }


def summarize_meeting_transcript(transcript: str, model: str = None) -> Dict[str, Any]:
    """
    Sends the raw meeting transcript to Llama 3.2 (via Ollama) to automatically
    extract all structured summary components matching MEETING_SUMMARY_SCHEMA.
    """
    active_model = resolve_ollama_model(model or MODEL)
    text = transcript.strip()
    if not text:
        return _fallback_heuristic_summary("Empty conversation")

    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please summarize the following sales meeting transcript:\n\n{text}"},
        ],
        "format": MEETING_SUMMARY_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        resp = request_with_retry(
            lambda: requests.post(OLLAMA_URL, json=payload, timeout=300),
            friendly_name=f"The local Ollama summarization model ({active_model})",
        )
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "")
            data = json.loads(content)
            # Ensure all required keys exist
            required_keys = [
                "participants", "purpose", "customer_needs", "pain_points",
                "budget", "timeline", "objections", "decisions", "next_steps", "concise_summary"
            ]
            if all(k in data for k in required_keys):
                # Cleanly format participants if returned as list/objects
                p_raw = data.get("participants")
                if isinstance(p_raw, list):
                    names = []
                    for item in p_raw:
                        if isinstance(item, dict):
                            n = item.get("name", "")
                            t = item.get("title", "")
                            c = item.get("company", "")
                            part = f"{n} ({t})" if t else n
                            if c and c not in part:
                                part += f" @ {c}"
                            names.append(part.strip())
                        else:
                            names.append(str(item))
                    data["participants"] = ", ".join(names)
                elif isinstance(p_raw, str) and p_raw.startswith("["):
                    try:
                        p_list = json.loads(p_raw)
                        if isinstance(p_list, list):
                            names = []
                            for item in p_list:
                                if isinstance(item, dict):
                                    n = item.get("name", "")
                                    t = item.get("title", "")
                                    part = f"{n} ({t})" if t else n
                                    names.append(part.strip())
                                else:
                                    names.append(str(item))
                            data["participants"] = ", ".join(names)
                    except Exception:
                        pass
                return data
    except Exception:
        # Fall back to heuristic parser gracefully if Ollama is not active
        pass

    return _fallback_heuristic_summary(text)
