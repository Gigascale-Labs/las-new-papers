"""Model calls, always with structured output.

Both calls in this pipeline want JSON, so there is one way to call a model
here: pass a JSON schema, get a checked dict back. The schema constrains
generation, so broken JSON is rare.

The retry still exists, for the cases a schema cannot cover: a refusal, a
response cut off at max_tokens, or a transport error.

Calls go through OpenRouter, not a single provider's API directly. `model` in
config.yaml is an OpenRouter model id (`anthropic/claude-opus-5`,
`openai/gpt-5`, any id OpenRouter lists). This module does not assume Claude.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)


class ModelError(Exception):
    """A model call failed twice, or failed in a way that cannot be retried."""


def _schema_name(label: str) -> str:
    """A name for the JSON schema, from the call's label.

    OpenRouter requires a-z, A-Z, 0-9, underscore, dash, max 64 chars. `label`
    is free text ("call 2 (2608.12345)"), so this strips what does not fit.
    """
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    return (name or "response")[:64]


class ModelClient:
    """Thin wrapper over the OpenRouter chat completions API.

    Small by design. The logic lives in the prompts, and a small surface lets
    the tests swap in a stub.
    """

    def __init__(self, model: str, effort: str = "medium", api_key: str | None = None):
        self.model = model
        self.effort = effort
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os

            from openrouter import OpenRouter    # imported late: not needed for --dry-run tests

            # No auto env resolution in this SDK, unlike some others. Read
            # OPENROUTER_API_KEY here if the caller did not pass a key.
            key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
            self._client = OpenRouter(api_key=key)
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
                return self._once(system, user, schema, max_tokens, label)
            except ModelError as exc:
                last_error = exc
                log.warning("%s: attempt %d/2 failed: %s", label, attempt, exc)
            except Exception as exc:
                last_error = exc
                log.warning("%s: attempt %d/2 errored: %s", label, attempt, exc)

        raise ModelError(f"{label}: failed twice: {last_error}")

    def _once(self, system: str, user: str, schema: dict, max_tokens: int, label: str) -> dict:
        from openrouter.errors import OpenRouterError

        try:
            resp = self._get_client().chat.send(
                model=self.model,
                max_tokens=max_tokens,
                reasoning_effort=self.effort,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": _schema_name(label), "schema": schema, "strict": True},
                },
            )
        except OpenRouterError as exc:
            raise ModelError(f"OpenRouter error: {exc}") from exc

        choice = resp.choices[0]

        if choice.finish_reason == "content_filter":
            raise ModelError("refused (content_filter)")
        if choice.finish_reason == "length":
            # Truncated JSON is unparseable; say so rather than let json.loads
            # report a confusing syntax error at the cut point.
            raise ModelError(f"response hit max_tokens ({max_tokens}); output truncated")
        if choice.finish_reason == "error":
            raise ModelError("provider reported an error")

        # Logged on every call, success or not: resp.id is the OpenRouter
        # generation id. Paste it into openrouter.ai/activity to see the exact
        # request and response OpenRouter recorded -- the one place the full,
        # un-truncated text is guaranteed to still exist after this call
        # returns. resp.model is which model actually served the request,
        # which can differ from the one asked for if OpenRouter routed to a
        # fallback.
        usage = getattr(resp, "usage", None)
        log.info(
            "%s: generation %s, served by %s, finish_reason=%s%s",
            label, getattr(resp, "id", "?"), getattr(resp, "model", "?"),
            choice.finish_reason,
            f", usage={usage}" if usage is not None else "",
        )

        text = choice.message.content
        if not text or not isinstance(text, str):
            raise ModelError("no text content in response")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError(f"invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ModelError(f"expected a JSON object, got {type(data).__name__}")

        # A schema-agnostic shape summary: works for call 1's {"scores": [...]}
        # and call 2's {"open_questions": [...], "canon": {...}} without this
        # module knowing either schema. This is what tells you "the call
        # succeeded but came back empty" apart from "the call came back with
        # the wrong shape entirely".
        shape = {
            k: f"list[{len(v)}]" if isinstance(v, list)
            else f"dict[{len(v)}]" if isinstance(v, dict)
            else type(v).__name__
            for k, v in data.items()
        }
        log.info("%s: response shape %s", label, shape)

        return data
