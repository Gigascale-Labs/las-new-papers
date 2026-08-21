"""Model calls, always with structured output.

Both calls in this pipeline want JSON back, so there is exactly one way to make
a call here: give it a JSON schema and get a validated dict. `output_config`'s
json_schema format constrains generation, so malformed JSON is close to
impossible -- but the spec's rule ("if the model returns bad JSON, ask once
more, then skip that paper") still has to hold for the cases that schema
enforcement does not cover: a refusal, a response truncated at max_tokens, or a
transport error.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


class ModelError(Exception):
    """A model call failed twice, or failed in a way that cannot be retried."""


class ModelClient:
    """Thin wrapper over the Anthropic Messages API.

    Kept small on purpose: the interesting logic is in the prompts, and a small
    surface is what lets the tests substitute a stub for it.
    """

    def __init__(self, model: str, effort: str = "medium", api_key: str | None = None):
        self.model = model
        self.effort = effort
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic                       # imported late: heavy, and not needed for --dry-run tests

            # No api_key argument means the SDK resolves ANTHROPIC_API_KEY, then
            # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 16000,
        label: str = "call",
    ) -> dict:
        """One call, one dict back. Retries once, then raises ModelError."""
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                return self._once(system, user, schema, max_tokens)
            except ModelError as exc:
                last_error = exc
                log.warning("%s: attempt %d/2 failed: %s", label, attempt, exc)
            except Exception as exc:
                last_error = exc
                log.warning("%s: attempt %d/2 errored: %s", label, attempt, exc)

        raise ModelError(f"{label}: failed twice: {last_error}")

    def _once(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        )

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise ModelError(f"refused ({getattr(details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            # Truncated JSON is unparseable; say so rather than let json.loads
            # report a confusing syntax error at the cut point.
            raise ModelError(f"response hit max_tokens ({max_tokens}); output truncated")

        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            raise ModelError("no text block in response")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ModelError(f"expected a JSON object, got {type(data).__name__}")
        return data
