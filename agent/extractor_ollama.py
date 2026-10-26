"""
Local, free extraction using Ollama -- no API key, no internet required
after setup, no per-call cost.

Setup (run once, in your terminal, NOT in Python):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.2
    3. Ollama runs a local server automatically at http://localhost:11434

Usage:
    from agent.extractor_ollama import extract_crm_data
    result = extract_crm_data("Had a call with Sarah Johnson...")
"""

import json
import logging
import requests
from typing import Optional

from agent.schema import CRM_EXTRACTION_TOOL
from agent.prompt_builder import build_system_prompt
from agent.http_utils import request_with_retry, FriendlyRequestError

logger = logging.getLogger("sales_meeting_agent.pipeline")

OLLAMA_URL = "http://localhost:11434/api/chat"
TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "llama3.2"

_SYSTEM_PROMPT = None
_CACHED_MODEL = None


def resolve_ollama_model(preferred: Optional[str] = "llama3.2") -> str:
    """
    Dynamically finds the best matching available Ollama model
    (e.g., 'llama3.2:3b', 'llama3.2', 'llama3:latest', etc.).
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and (preferred is None or preferred == DEFAULT_MODEL):
        return _CACHED_MODEL

    target = preferred or DEFAULT_MODEL
    try:
        resp = requests.get(TAGS_URL, timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            if target in models:
                _CACHED_MODEL = target
                return target
            for m in models:
                if m.startswith(target + ":") or target in m:
                    _CACHED_MODEL = m
                    return m
            for prefix in ["llama3.2", "llama3", "llama", "gemma"]:
                for m in models:
                    if prefix in m.lower():
                        _CACHED_MODEL = m
                        return m
            if models:
                _CACHED_MODEL = models[0]
                return models[0]
    except Exception:
        pass

    _CACHED_MODEL = target
    return target


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
    active_model = resolve_ollama_model(model or DEFAULT_MODEL)
    logger.info("Sending transcript to CRM extractor")

    payload = {
        "model": active_model,
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
            friendly_name=f"The local Ollama model ({active_model})",
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
