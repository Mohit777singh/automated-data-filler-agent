"""
Full end-to-end AI Sales Meeting Agent flow.
Automatically summarizes raw meeting transcripts/conversations into participants,
purpose, needs, pain points, budget, timeline, objections, decisions, and next steps,
extracts CRM records with Llama 3.2, and formulates actions with human-in-the-loop confirmation.

Run from the project root:
    python -m agent.run_agent
"""

import sys
import json

# Ensure stdout handles unicode/emojis gracefully across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent.sales_meeting_agent import SalesMeetingAgent, AgentResult
from agent.mock_crm_client import MockCRMClient


def print_agent_output(result: AgentResult):
    """Prints a structured breakdown of the agent's complete intelligence report."""
    crm = result.crm_data
    c, a, o = crm.get("Contact", {}), crm.get("Account", {}), crm.get("Opportunity", {})
    summary = result.summary

    print("\n" + "=" * 75)
    print("AI SALES MEETING AGENT - EXECUTIVE INTELLIGENCE & AUTOMATIC SUMMARY")
    print("=" * 75)

    # 1. Automatic Meeting Summary
    print("\n[1] AUTOMATIC MEETING SUMMARY (GENERATED FROM TRANSCRIPT)")
    print(f"  • Participants:   {summary.get('participants')}")
    print(f"  • Purpose:        {summary.get('purpose')}")
    print(f"  • Budget:         {summary.get('budget')}")
    print(f"  • Timeline:       {summary.get('timeline')}")
    
    print("  • Customer Needs:")
    for need in summary.get("customer_needs", []):
        print(f"     + {need}")

    print("  • Core Pain Points:")
    for pp in summary.get("pain_points", []):
        print(f"     ! {pp}")

    if summary.get("objections"):
        print("  • Objections / Alternatives:")
        for obj in summary.get("objections", []):
            print(f"     - {obj}")

    if summary.get("decisions"):
        print("  • Decisions Reached:")
        for dec in summary.get("decisions", []):
            print(f"     ✓ {dec}")

    print("  • Next Steps:")
    for nxt in summary.get("next_steps", []):
        print(f"     -> {nxt}")

    print(f"\n  Executive Narrative:\n  \"{summary.get('concise_summary')}\"")

    # 2. Conversation Analysis
    analysis = result.analysis
    print("\n[2] CONVERSATION ANALYSIS & SIGNALS")
    print(f"  • Meeting Type:   {analysis.get('meeting_type')}")
    print(f"  • Buyer Tone:     {analysis.get('sentiment')}")
    print(f"  • Buying Intent:  {analysis.get('buying_intent')}")
    print(f"  • Urgency Level:  {analysis.get('urgency')}")
    if analysis.get("competitors_mentioned"):
        print(f"  • Competitors:    {', '.join(analysis['competitors_mentioned'])}")

    # 3. Information Need Assessment
    info = result.information_needed
    print("\n[3] INFORMATION NEED ASSESSMENT")
    print(f"  • Completeness:   {info.get('completeness_score')}% ({info.get('status')})")
    print("  • Captured Data:")
    for cap in info.get("captured_fields", []):
        print(f"     [+] {cap}")
    if info.get("missing_critical_fields"):
        print("  • Missing / Requires Discovery:")
        for mis in info.get("missing_critical_fields", []):
            print(f"     [!] {mis}")
    if info.get("recommended_discovery_questions"):
        print("  • Recommended Rep Questions:")
        for q in info.get("recommended_discovery_questions", []):
            print(f"     (?) \"{q}\"")

    # 4. CRM Extraction
    print("\n[4] EXTRACTED CRM DATA")
    print("  Contact:")
    print(f"    Name:  {c.get('FirstName') or ''} {c.get('LastName') or ''}".strip() or "    (none found)")
    if c.get("Title"):
        print(f"    Title: {c['Title']}")
    if c.get("Email"):
        print(f"    Email: {c['Email']}")

    print("  Account:")
    print(f"    Company:   {a.get('Name') or '(none found)'}")
    if a.get("Industry"):
        print(f"    Industry:  {a['Industry']}")
    if a.get("NumberOfEmployees"):
        print(f"    Team Size: {a['NumberOfEmployees']} employees")
    if a.get("BudgetRange__c"):
        print(f"    Budget:    {a['BudgetRange__c']}")

    print("  Opportunity:")
    print(f"    Deal:      {o.get('Name') or '(none found)'}")
    if o.get("Amount"):
        print(f"    Value:     ${o['Amount']:,.0f}")
    print(f"    Stage:     {o.get('StageName') or 'Prospecting'}")
    if o.get("Competitor__c"):
        print(f"    Competitor:{o['Competitor__c']}")
    if o.get("NextStep"):
        print(f"    Next Step: {o['NextStep']}")
    print(f"  Confidence:  {crm.get('confidence', 'unknown').title()}")

    # 5. Missed Questions & What We Should Have Said
    coaching = result.coaching
    if coaching:
        print("\n[5] 🎓 MISSED QUESTIONS & WHAT WE SHOULD HAVE SAID")
        print(f"  • Sales Execution Score: {coaching.get('overall_performance_score', 80)} / 100")
        
        if coaching.get("missed_questions") or coaching.get("failed_to_ask"):
            print("\n  ❌ What the Salesperson Failed to Ask (Missed Questions):")
            for f in (coaching.get("missed_questions") or coaching.get("failed_to_ask", [])):
                print(f"     - {f}")

        if coaching.get("what_we_should_have_said"):
            print("\n  💬 What We Should Have Said (Ideal Scripted Responses):")
            for said in coaching.get("what_we_should_have_said", []):
                print(f"     👉 {said}")

        if coaching.get("recommended_questions_to_ask"):
            print("\n  📋 Tailored Question Scripts & Rationale:")
            for q in coaching.get("recommended_questions_to_ask", []):
                print(f"     • Context:   {q.get('context')}")
                print(f"       Script:    {q.get('recommended_question')}")
                print(f"       Rationale: {q.get('rationale')}\n")

        print(f"  Coaching Takeaway:\n  \"{coaching.get('coaching_takeaway')}\"")

    # 6. Next Best Action (NBA)
    actions = result.actions
    nba = actions.get("next_best_action", {})
    print("\n[6] 🎯 NEXT BEST ACTION (NBA) & STRATEGIC RATIONALE")
    if nba:
        print(f"  • PRIMARY ACTION:  {nba.get('primary_action')}")
        print(f"  • STRATEGIC WHY:   {nba.get('rationale')}")
        print(f"  • URGENCY:         {nba.get('urgency')}")
        print(f"  • TARGET MILESTONE:{nba.get('target_milestone')}")
        if nba.get("deal_acceleration_strategy"):
            print(f"  • ACCELERATION:    {nba.get('deal_acceleration_strategy')}")

    # 7. Action Items & Follow-Ups
    print("\n[7] ⚡ ACTION ITEMS & CRM DATABASE WRITE")
    print("  Required Follow-Up Sequence:")
    for fu in (nba.get("required_follow_ups") or actions.get("tasks", [])):
        cat = fu.get('category') or f"{fu.get('priority', 'High')} Priority"
        act_title = fu.get('action') or fu.get('title')
        owner = fu.get('suggested_owner')
        deadline = fu.get('deadline') or fu.get('due_timeline')
        print(f"    -> [{cat}] {act_title} (Owner: {owner} | Target: {deadline})")

    print("\n  Planned CRM Record Ops:")
    for act in actions.get("crm_actions", []):
        print(f"    • {act}")

    if actions.get("risk_alerts"):
        print("\n  Risk Alerts:")
        for rk in actions.get("risk_alerts", []):
            print(f"    [!] {rk}")

    print("\n  AI-Drafted Contextual Follow-Up Email:")
    print("  " + "-" * 50)
    email = actions.get("follow_up_email", {})
    print(f"  Subject: {email.get('subject')}\n")
    for line in email.get("body", "").split("\n"):
        print(f"  {line}")
    print("  " + "-" * 50)
    print("=" * 75)



def run_pipeline(meeting_transcript: str, auto_confirm: bool = True, model_name: str = "llama3.2"):
    """
    Runs the full AI Sales Meeting Agent pipeline:
    Auto-Summarize -> Analyze -> Assess Info Needs -> Extract CRM ->
    Missed Questions -> Next Best Action -> Automated CRM Execution (Contact, Account, Opportunity, Tasks).
    """
    print("Initializing AI Sales Meeting Agent...")
    print(f"\n> Raw Input Transcript / Conversation:\n{meeting_transcript}\n")

    agent = SalesMeetingAgent(model_name=model_name)

    print("Agent is analyzing meeting, determining CRM operations, missed questions, Next Best Action, and action items...")
    try:
        result = agent.run_end_to_end_pipeline(meeting_transcript, auto_confirm=auto_confirm)
    except Exception as e:
        print(f"\n[Agent Error] Failed to process meeting: {e}")
        return None

    print_agent_output(result)

    if result.validation_warnings:
        print("\nExtraction Warnings:")
        for w in result.validation_warnings:
            print(f"  - {w}")

    write_res = result.crm_write_result or {}
    if write_res:
        print("\n===========================================================================")
        print("✅ AUTOMATED CRM OPERATIONS EXECUTED")
        print("===========================================================================")
        for key in ("account", "contact", "opportunity"):
            record_id = write_res.get(f"{key}_id")
            action = write_res.get(f"{key}_action")
            if record_id:
                print(f"  • {key.capitalize():12}: [{action.upper()}] ID: {record_id}")
            else:
                print(f"  • {key.capitalize():12}: not created (no data)")

        tasks_created = write_res.get("tasks_created", 0)
        tasks_updated = write_res.get("tasks_updated", 0)
        task_ids = write_res.get("task_ids", [])
        print(f"  • Action Items: {tasks_created} created, {tasks_updated} updated (Total Task IDs: {len(task_ids)})")
        print("===========================================================================")

    return result


SAMPLE_TRANSCRIPT = (
    "Alex (AE): Hi Sarah, great to connect with you. Thanks for joining today's discovery call.\n"
    "Sarah Johnson (Marketing Director @ GrowthTech): Hi Alex! Thanks for having me. We are a 50-person SaaS company.\n"
    "Alex (AE): What's the main challenge you're looking to solve?\n"
    "Sarah Johnson: Our team is wasting hours manually sending onboarding emails and updating lists. We need automated marketing workflows and a solid reporting suite.\n"
    "Alex (AE): Got it. Have you established a budget for this platform?\n"
    "Sarah Johnson: Yes, our VP approved an annual budget around $30,000.\n"
    "Alex (AE): Are you evaluating other tools currently?\n"
    "Sarah Johnson: We looked at HubSpot, but their reporting is rigid. Your analytics look much stronger.\n"
    "Alex (AE): Wonderful. Let's schedule a dedicated ops walkthrough for your team this Friday.\n"
    "Sarah Johnson: Friday works perfectly for us. Let's do it."
)

if __name__ == "__main__":
    run_pipeline(SAMPLE_TRANSCRIPT)

