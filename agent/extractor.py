"""
Core extraction function for the Automated Data Filler Agent.

Usage (from project root):
    export ANTHROPIC_API_KEY=your_key_here      # or put it in a .env file
    python3 -m agent.extractor

Or import and call from a notebook:
    from agent.extractor import extract_crm_data
    result = extract_crm_data("Had a call with Sarah Johnson...")
"""

import os
import json
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up ANTHROPIC_API_KEY from a .env file if present
except ImportError:
    pass  # dotenv is optional -- env var can also be set directly

from agent.schema import CRM_EXTRACTION_TOOL
from agent.prompt_builder import build_system_prompt
from agent.http_utils import request_with_retry, FriendlyRequestError

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-5"  # swap for whichever current model string you're using

_SYSTEM_PROMPT = None  # cached after first build


def _get_system_prompt():
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = build_system_prompt()
    return _SYSTEM_PROMPT


def extract_crm_data(meeting_summary: str, api_key: str = None) -> dict:
    """
    Sends a meeting summary to Claude and returns structured CRM data
    matching the Salesforce-style schema. Raises RuntimeError on API
    or parsing failure -- callers should catch this and handle it
    (e.g. surface an error to the user rather than writing garbage to CRM).
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY in your environment "
            "or a .env file, or pass api_key= directly."
        )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 1500,
        "system": _get_system_prompt(),
        "messages": [{"role": "user", "content": meeting_summary}],
        "tools": [CRM_EXTRACTION_TOOL],
        "tool_choice": {"type": "tool", "name": "extract_crm_data"},
    }

    try:
        resp = request_with_retry(
            lambda: requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30),
            friendly_name="The Claude API",
        )
    except FriendlyRequestError as e:
        raise RuntimeError(str(e))

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()

    tool_use_block = next(
        (block for block in data.get("content", []) if block.get("type") == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise RuntimeError(f"No tool_use block in response: {data}")

    return tool_use_block["input"]


if __name__ == "__main__":
    test_summary = (
        "Had a call with Sarah Johnson, Marketing Director at GrowthTech. "
        "They need marketing automation for 50-person team. Budget: $30K annually. "
        "Evaluating HubSpot vs our solution. Next: Demo scheduled for Friday."
    )
    try:
        result = extract_crm_data(test_summary)
        print(json.dumps(result, indent=2))
    except RuntimeError as e:
        print(f"[extraction failed] {e}")
