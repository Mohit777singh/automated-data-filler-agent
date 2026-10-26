"""
Hackathon-ready prompt builder for CRM extraction.

Improvements:
- Strict anti-hallucination rules
- Deterministic few-shot examples (stable demos)
- Per-field evidence extraction
- Review-required flag
- Negative example to prevent guessing
- Enterprise-style confidence handling
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "merged_train.jsonl"

SYSTEM_PROMPT_TEMPLATE = """
You are an enterprise CRM data extraction assistant for a B2B sales team.

Your task is to convert a sales meeting summary into structured CRM data.

========================
CRITICAL RULES
========================

1. Extract ONLY information explicitly stated in the meeting summary.
2. Never infer, estimate, or assume values.
3. If a field is not mentioned, return null.
4. Copy names, companies, and titles exactly as written.
5. Opportunity.Amount must be a plain number (30000, not "$30K").

========================
NEVER INFER
========================

- Industry
- Number of employees
- Website
- Email address
- Phone number
- Budget
- Deal stage
- Competitor
- Close date

If any of these are not explicitly mentioned, return null.

========================
CONFIDENCE
========================

- high: all important extracted fields are explicitly stated.
- medium: some fields are explicit but one or more fields are ambiguous.
- low: little or no CRM-relevant information is present.

========================
STAGE DETECTION
========================

Do NOT default to "Negotiation/Review" when the stage is unclear. Use the
EARLIEST defensible stage the conversation gives clear evidence for:

- "Prospecting": initial contact, no clear need identified yet.
- "Qualification" / "Needs Analysis": customer describes a need or pain point.
- "Value Proposition": you've described how your product helps.
- "Proposal/Price Quote": a proposal, quote, or pricing was sent or requested.
- "Negotiation/Review": pricing, terms, or contract are ACTIVELY being
  negotiated -- not just mentioned.
- "Demo Scheduled": a demo has an explicit date/time set.
- "Closed Won" / "Closed Lost": the customer explicitly agreed or declined.

If the conversation doesn't clearly support any of these, use "Prospecting"
rather than guessing a more advanced stage.

========================
INSIGHTS EXTRACTION
========================

In addition to the core Contact/Account/Opportunity fields, extract an
"Insights" object with: ProductOrPlan, UsersOrLicenses, Requirements,
InterestedFeatures, PainPoints, Objections, CompetitorsMentioned, Sentiment,
BuyingIntent, ExpectedDecisionDate, FollowUpDate.

Rules for Insights fields specifically (different from the null-based CRM
fields above):
- String fields (ProductOrPlan, UsersOrLicenses, ExpectedDecisionDate,
  FollowUpDate): use the literal string "Unknown" if not stated.
- List fields (Requirements, InterestedFeatures, PainPoints, Objections,
  CompetitorsMentioned): use an empty list [] if nothing is stated. Only
  include items explicitly present in the conversation.
- Sentiment: one of Positive / Neutral / Negative / Mixed / Unknown --
  base this on the customer's tone, not the salesperson's.
- BuyingIntent: one of High / Medium / Low / Unknown -- base this on
  concrete signals (e.g. "let's move forward" = High; "we'll think about
  it" = Low/Medium), not on Opportunity.StageName.
- Never invent a competitor, feature, or pain point that isn't stated.
  If the customer mentions frustration or a problem but doesn't name a
  specific pain point, it's fine to paraphrase closely, but don't invent
  specifics (dollar figures, product names) that weren't said.

========================
REVIEW REQUIRED
========================

Set review_required = true if any extracted value is uncertain,
ambiguous, or only partially supported by the text.

========================
EVIDENCE
========================

For every non-null field, include a short supporting phrase copied
directly from the meeting summary.

Example:
"evidence": {{
  "Contact.Title": "Founder of UrbanCart",
  "Opportunity.Amount": "Budget is $2,500 per month"
}}

========================
ANTI-HALLUCINATION EXAMPLE
========================

Meeting:
"Spoke with Priya from UrbanCart."

WRONG:
{{
  "Industry": "E-commerce",
  "NumberOfEmployees": 1000
}}

CORRECT:
{{
  "Industry": null,
  "NumberOfEmployees": null
}}

========================
FEW-SHOT EXAMPLES
========================

{few_shot_examples}

========================
FINAL INSTRUCTION
========================

Return only valid JSON matching the CRM schema.
Do not include explanations, markdown, or extra text.
"""


def format_example(row, index):
    summary = row["meeting_summary"]
    record = row["salesforce_record"]

    return (
        f"Example {index}:\\n"
        f"Meeting summary:\\n{summary}\\n\\n"
        f"Correct extraction:\\n{json.dumps(record, indent=2)}\\n"
    )


def build_system_prompt(train_path=None):
    """
    Deterministic few-shot selection for stable hackathon demos.
    Picks:
    - 2 fully populated examples
    - 1 partial-information example
    - 1 sparse/no-CRM example
    """

    train_path = Path(train_path) if train_path else DEFAULT_TRAIN_PATH

    with open(train_path) as f:
        rows = [json.loads(line) for line in f]

    synthetic_rows = [r for r in rows if r.get("source") == "synthetic"]
    sparse_rows = [r for r in rows if r.get("source") == "sales_conversations"]

    # Stable selection (no randomness during demos)
    selected = []

    # Two strong examples
    selected.extend(synthetic_rows[:2])

    # One partial-information example if available
    if len(synthetic_rows) >= 3:
        selected.append(synthetic_rows[2])

    # One sparse/no-business-content example
    if sparse_rows:
        selected.append(sparse_rows[0])

    examples_text = "\\n".join(
        format_example(r, i + 1) for i, r in enumerate(selected)
    )

    return SYSTEM_PROMPT_TEMPLATE.format(few_shot_examples=examples_text)


if __name__ == "__main__":
    print(build_system_prompt())