"""
Shared HTTP retry helper. Wraps a single request-making function with a
"retry once on 5xx / connection error" pattern, since transient failures
(model server briefly overloaded, momentary network blip) shouldn't
immediately surface as a hard crash to the end user.

Usage:
    from agent.http_utils import request_with_retry

    resp = request_with_retry(
        lambda: requests.post(url, json=payload, timeout=30),
        friendly_name="Ollama extraction",
    )
"""

import time
import requests


class FriendlyRequestError(RuntimeError):
    """Raised with a clear, user-facing message after retry has been exhausted."""
    pass


def request_with_retry(make_request, friendly_name: str, retry_delay_seconds: float = 2.0):
    """
    Calls make_request() (a zero-arg callable that performs one HTTP call
    and returns the Response). Retries exactly once if the first attempt
    fails with a 5xx status or a connection/timeout error. Raises a
    FriendlyRequestError with a clear message if both attempts fail.
    """
    last_error_message = None

    for attempt in (1, 2):
        try:
            resp = make_request()
        except requests.exceptions.ConnectionError:
            last_error_message = f"{friendly_name} is unreachable (connection failed)."
        except requests.exceptions.Timeout:
            last_error_message = f"{friendly_name} timed out."
        else:
            if resp.status_code < 500:
                # Success, or a non-retryable client error (4xx) -- return
                # immediately either way; 4xx errors are the caller's to
                # interpret (e.g. bad auth), not something a retry fixes.
                return resp
            last_error_message = f"{friendly_name} returned a server error ({resp.status_code})."

        if attempt == 1:
            time.sleep(retry_delay_seconds)

    raise FriendlyRequestError(
        f"{last_error_message} Your input wasn't lost -- you can try again. "
        f"If this keeps happening, check that the service is running/reachable."
    )
