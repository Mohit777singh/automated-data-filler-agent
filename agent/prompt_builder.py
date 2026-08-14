"""
Builds the system prompt for extraction, including few-shot examples
pulled directly from data/merged_train.jsonl:
  - a few 'synthetic' rows (fully labeled) to show correct extraction
  - one 'sales_conversations' row (sparse/no CRM info) to demonstrate
    correct null-handling instead of hallucination
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "merged_train.jsonl"

SYSTEM_PROMPT_TEMPLATE = """You are a CRM data extraction assistant for a B2B sales team. \
Your job is to read a sales meeting summary and extract structured CRM data.

Rules:
1. Only extract information that is explicitly stated or very clearly implied.
2. If a field is not mentioned in the text, its value MUST be null. Never guess, \
never infer a plausible-sounding value, never fill in a "reasonable default".
3. Names, companies, and titles should be extracted exactly as written.
4. Dollar amounts go in Opportunity.Amount as a plain number (e.g. 30000, not "$30K").
5. If the text contains no business/sales content at all (e.g. casual conversation \
with no company, deal, or contact info), return an object with every field null \
and confidence "low".

Here are examples of correct extraction:

{few_shot_examples}

Now extract CRM data from the meeting summary the user provides. Call the \
extract_crm_data tool with your result.
"""


def format_example(row, index):
    summary = row["meeting_summary"]
    record = row["salesforce_record"]
    return (
        f"Example {index}:\n"
        f"Meeting summary: \"{summary}\"\n"
        f"Correct extraction:\n{json.dumps(record, indent=2)}\n"
    )


def build_system_prompt(train_path=None, seed=3):
    """
    Selects a small, diverse set of few-shot examples:
    - 3 synthetic rows (fully labeled, varied completeness)
    - 1 sales_conversations row (sparse -- teaches correct null behavior)
    """
    train_path = Path(train_path) if train_path else DEFAULT_TRAIN_PATH
    random.seed(seed)
    with open(train_path) as f:
        rows = [json.loads(line) for line in f]

    synthetic_rows = [r for r in rows if r["source"] == "synthetic"]
    sparse_rows = [r for r in rows if r["source"] == "sales_conversations"]

    few_shot = random.sample(synthetic_rows, 3) + random.sample(sparse_rows, 1)
    random.shuffle(few_shot)

    examples_text = "\n".join(format_example(r, i + 1) for i, r in enumerate(few_shot))
    return SYSTEM_PROMPT_TEMPLATE.format(few_shot_examples=examples_text)


if __name__ == "__main__":
    print(build_system_prompt())
