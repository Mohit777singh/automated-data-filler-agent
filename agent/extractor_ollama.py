"""
Local, free extraction using Ollama -- no API key, no internet required
after setup, no per-call cost.

Setup (run once, in your terminal, NOT in Python):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.2
       (smaller/faster alternative for lower-spec machines: ollama pull llama3.2)
    3. Ollama runs a local server automatically at http://localhost:11434
       -- just leave the Ollama app open in the background.

Usage:
    from agent.extractor_ollama import extract_crm_data
    result = extract_crm_data("Had a call with Sarah Johnson...")
"""

import json
import requests

from agent.schema import CRM_EXTRACTION_TOOL
from agent.prompt_builder import build_system_prompt
from agent.http_utils import request_with_retry, FriendlyRequestError

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # change to "llama3.2" or another pulled model if needed

_SYSTEM_PROMPT = None


def _get_system_prompt():
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = build_system_prompt()
    return _SYSTEM_PROMPT


def extract_crm_data(meeting_summary: str, model: str = None) -> dict:
    """
    Sends a meeting summary to a locally-running Ollama model and returns
    structured CRM data matching the same schema used by the Claude-based
    extractor. Uses Ollama's structured-output "format" parameter (a JSON
    schema) to force valid, schema-matching JSON back.
    """
    model = model or MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": meeting_summary},
        ],
        "format": CRM_EXTRACTION_TOOL["input_schema"],  # forces structured JSON output
        "stream": False,
        "options": {"temperature": 0},  # deterministic extraction
    }

    try:
        resp = request_with_retry(
            lambda: requests.post(OLLAMA_URL, json=payload, timeout=300),
            friendly_name="The local Ollama model",
        )
    except FriendlyRequestError as e:
        raise RuntimeError(
            f"{e} (Is the Ollama app running? Open it and try again.)"
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data.get("message", {}).get("content", "")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw output: {content}")


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
