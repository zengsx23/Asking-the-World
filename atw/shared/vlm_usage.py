"""Automatic OpenRouter usage and USD-cost accounting for VLM modules."""

from __future__ import annotations

from pathlib import Path

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from threading import Lock
from contextvars import ContextVar
from typing import Any
import math
import re


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]]
REQUEST_RETRY_FACTORY = ContextVar("vlm_request_retry_factory", default=None)


class VLMRequestRetries:
    """Per-step harness retries, independent of scene correction budgets."""

    def __init__(self, path):
        self.path = path
        self.context = {}
        self.counts = dict(request_errors=0, request_retries=0, format_errors=0, format_retries=0)

    def retry(self, kind, error):
        from atw.perception.pipeline.recording import write_json

        self.counts[kind + "_errors"] += 1
        allowed = kind == "request" or self.counts["format_retries"] < 2
        if allowed:
            self.counts[kind + "_retries"] += 1
        write_json(self.path, dict(self.context, **self.counts, last_error_type=type(error).__name__,
                                  last_error=str(error), retry_allowed=allowed))
        if allowed and kind == "request":
            time.sleep(min(30, 2 ** min(self.counts["request_retries"], 5)))
        return allowed


def default_openrouter_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("OpenRouter returned a non-object response")
    return data


def extract_openrouter_json_response(
    response: Mapping[str, Any], *, allow_single_object_array: bool = False,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise TypeError("OpenRouter choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("OpenRouter choice has no message object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned empty message content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    parsed = json.loads(text)
    if allow_single_object_array and isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise TypeError("VLM response must be a JSON object")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sum_int(calls: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(
        value
        for call in calls
        if isinstance((value := call.get(field)), int) and not isinstance(value, bool)
    )


def _sum_decimal(calls: Sequence[Mapping[str, Any]], field: str) -> float:
    total = Decimal("0")
    for call in calls:
        value = call.get(field)
        if value is None or isinstance(value, bool):
            continue
        try:
            total += Decimal(str(value))
        except Exception:
            continue
    return float(total)


def estimate_text_tokens(payload: Mapping[str, Any]) -> int:
    """Conservative request estimate excluding encoded media bytes."""
    def scrub(value):
        if isinstance(value, Mapping):
            if value.get("type") in {"image_url", "video_url", "audio_url", "input_image", "input_video"}:
                return {"type": value.get("type"), "media": "<omitted>"}
            return {k: scrub(v) for k, v in value.items()
                    if k not in {"model", "session_id"}}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str) and value.startswith("data:"):
            return "<media>"
        return value
    text = json.dumps(scrub(payload), ensure_ascii=False, separators=(",", ":"))
    cjk = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    return cjk + math.ceil((len(text) - cjk) / 4)


def _summarize_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    responses = [call for call in calls if call.get("response_received") is True]
    return {
        "attempt_count": len(calls),
        "response_count": len(responses),
        "accounted_response_count": sum(
            call.get("usage_available") is True for call in responses
        ),
        "missing_usage_response_count": sum(
            call.get("usage_available") is not True for call in responses
        ),
        "missing_cost_response_count": sum(call.get("cost_usd") is None for call in responses),
        "cost_complete": all(call.get("cost_usd") is not None for call in responses)
        and len(responses) == len(calls),
        "transport_failure_count": sum(
            call.get("response_received") is not True for call in calls
        ),
        "cache_hit_response_count": sum(
            isinstance(call.get("cached_tokens"), int)
            and int(call["cached_tokens"]) > 0
            for call in responses
        ),
        "prompt_tokens": _sum_int(responses, "prompt_tokens"),
        "completion_tokens": _sum_int(responses, "completion_tokens"),
        "reasoning_tokens": _sum_int(responses, "reasoning_tokens"),
        "cached_tokens": _sum_int(responses, "cached_tokens"),
        "cache_write_tokens": _sum_int(responses, "cache_write_tokens"),
        "estimated_text_tokens": _sum_int(calls, "estimated_text_tokens"),
        "image_tokens": _sum_int(responses, "image_tokens"),
        "video_tokens": _sum_int(responses, "video_tokens"),
        "audio_tokens": _sum_int(responses, "audio_tokens"),
        "visual_tokens": _sum_int(responses, "visual_tokens"),
        "total_tokens": _sum_int(responses, "total_tokens"),
        "cost_usd": _sum_decimal(responses, "cost_usd"),
        "upstream_inference_cost_usd": _sum_decimal(
            responses, "upstream_inference_cost_usd"
        ),
    }


class VLMUsageLedger:
    """Thread-safe per-module ledger populated from provider response metadata."""

    schema_version = 1

    def __init__(self, module_name: str) -> None:
        if not module_name.strip():
            raise ValueError("module_name cannot be empty")
        self.module_name = module_name
        self._calls: list[dict[str, Any]] = []
        self._lock = Lock()

    def record_response(
        self,
        *,
        operation: str,
        call_id: str,
        attempt: int,
        requested_model: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        usage = response.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else None
        prompt_details: Mapping[str, Any] = {}
        completion_details: Mapping[str, Any] = {}
        cost_details: Mapping[str, Any] = {}
        if usage_mapping is not None:
            raw_prompt_details = usage_mapping.get("prompt_tokens_details")
            raw_completion_details = usage_mapping.get("completion_tokens_details")
            raw_cost_details = usage_mapping.get("cost_details")
            if isinstance(raw_prompt_details, Mapping):
                prompt_details = raw_prompt_details
            if isinstance(raw_completion_details, Mapping):
                completion_details = raw_completion_details
            if isinstance(raw_cost_details, Mapping):
                cost_details = raw_cost_details
        record = {
            "module": self.module_name,
            "operation": operation,
            "call_id": call_id,
            "attempt": attempt,
            "response_received": True,
            "json_parse_succeeded": False,
            "generation_id": (
                str(response["id"]) if response.get("id") is not None else None
            ),
            "requested_model": requested_model,
            "response_model": (
                str(response["model"]) if response.get("model") is not None else None
            ),
            "provider": (
                str(response["provider"])
                if response.get("provider") is not None
                else None
            ),
            "usage_available": usage_mapping is not None,
            "prompt_tokens": _optional_int(
                usage_mapping.get("prompt_tokens") if usage_mapping else None
            ),
            "completion_tokens": _optional_int(
                usage_mapping.get("completion_tokens") if usage_mapping else None
            ),
            "reasoning_tokens": _optional_int(
                completion_details.get("reasoning_tokens")
            ),
            "cached_tokens": _optional_int(prompt_details.get("cached_tokens")),
            "cache_write_tokens": _optional_int(
                prompt_details.get("cache_write_tokens")
            ),
            "image_tokens": _optional_int(prompt_details.get("image_tokens")),
            "video_tokens": _optional_int(prompt_details.get("video_tokens")),
            "audio_tokens": _optional_int(prompt_details.get("audio_tokens")),
            "total_tokens": _optional_int(
                usage_mapping.get("total_tokens") if usage_mapping else None
            ),
            "cost_usd": _optional_float(
                usage_mapping.get("cost") if usage_mapping else None
            ),
            "upstream_inference_cost_usd": _optional_float(
                cost_details.get("upstream_inference_cost")
            ),
        }
        visual = [record[k] for k in ("image_tokens", "video_tokens")
                  if isinstance(record[k], int)]
        record["visual_tokens"] = sum(visual) if visual else None
        record["visual_token_details_available"] = bool(visual)
        with self._lock:
            self._calls.append(record)
        return record

    def record_transport_failure(
        self,
        *,
        operation: str,
        call_id: str,
        attempt: int,
        requested_model: str,
        error: Exception,
    ) -> dict[str, Any]:
        record = {
            "module": self.module_name,
            "operation": operation,
            "call_id": call_id,
            "attempt": attempt,
            "response_received": False,
            "json_parse_succeeded": False,
            "generation_id": None,
            "requested_model": requested_model,
            "response_model": None,
            "provider": None,
            "usage_available": False,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "cache_write_tokens": None,
            "image_tokens": None,
            "video_tokens": None,
            "audio_tokens": None,
            "visual_tokens": None,
            "visual_token_details_available": False,
            "total_tokens": None,
            "cost_usd": None,
            "upstream_inference_cost_usd": None,
            "error_type": type(error).__name__,
        }
        with self._lock:
            self._calls.append(record)
        return record

    def report(self) -> dict[str, Any]:
        with self._lock:
            calls = [dict(call) for call in self._calls]
        return {
            "schema_version": self.schema_version,
            "currency": "USD",
            "module": self.module_name,
            "totals": _summarize_calls(calls),
            "calls": calls,
        }


class OpenRouterJSONClient:
    """Shared JSON client that accounts for every OpenRouter response before parsing."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int,
        module_name: str,
        transport: Transport | None = None,
        max_attempts: int = 3,
        trace_dir: Path | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.api_key = api_key
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.transport = transport or default_openrouter_transport
        self.max_attempts = max_attempts
        self.trace_dir = trace_dir
        self.usage = VLMUsageLedger(module_name)

    def request_json(
        self,
        *,
        operation: str,
        call_id: str,
        system: str,
        content: list[dict[str, Any]],
        object_id: str | None = None,
        retry_policy: VLMRequestRetries | None = None,
        validate: Callable | None = None,
        allow_single_object_array: bool = False,
    ) -> dict[str, Any]:
        from atw.perception.pipeline.recording import CURRENT

        recorder = CURRENT.get()
        factory = REQUEST_RETRY_FACTORY.get()
        if factory is not None:
            retry_policy = factory()
            allow_single_object_array = True
        elif retry_policy is None and recorder is not None:
            retry_policy = recorder.vlm_retry_policy()
        if retry_policy is not None:
            retry_policy.context = dict(module=self.usage.module_name, operation=operation, call_id=call_id)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        attempt = 0
        while True:
            attempt += 1
            self.last_raw_response = None
            last_error = None
            started = time.perf_counter()
            record = None
            error_kind = "request"
            parsed = None
            trace = None
            estimated_text_tokens = estimate_text_tokens(payload)
            if self.trace_dir is not None:
                from atw.shared.request_trace import request_trace
                trace = request_trace(self.trace_dir, operation, call_id, attempt, payload)
            try:
                response = self.transport(OPENROUTER_CHAT_URL, headers, payload)
            except (OSError, urllib.error.URLError, ValueError, TypeError) as exc:
                record = self.usage.record_transport_failure(
                    operation=operation,
                    call_id=call_id,
                    attempt=attempt,
                    requested_model=self.model,
                    error=exc,
                )
                last_error = exc
            else:
                self.last_raw_response = response
                record = self.usage.record_response(
                    operation=operation,
                    call_id=call_id,
                    attempt=attempt,
                    requested_model=self.model,
                    response=response,
                )
                try:
                    error_kind = "request"
                    provider_error = response.get("error")
                    if provider_error:
                        raise OSError(f"Provider request error: {provider_error}")
                    for choice in response.get("choices") or []:
                        if isinstance(choice, Mapping) and (choice.get("error") or choice.get("finish_reason") in ("error", "length", "content_filter")):
                            raise OSError(f"Provider request error: {choice.get('error') or 'finish_reason=error'}")
                    error_kind = "format"
                    parsed = extract_openrouter_json_response(response, allow_single_object_array=allow_single_object_array)
                    record["json_parse_succeeded"] = True
                    if validate is not None:
                        validate(parsed)
                    return parsed
                except (OSError, KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    last_error = exc
            finally:
                if record is not None:
                    from atw.perception.pipeline.recording import CURRENT
                    record["response_seconds"] = time.perf_counter() - started
                    record["object_id"] = object_id
                    record["estimated_text_tokens"] = estimated_text_tokens
                    recorder = CURRENT.get()
                    if recorder is not None:
                        recorder.vlm_call(record)
                    if trace is not None:
                        from atw.shared.request_trace import response_trace
                        response_trace(trace, self.last_raw_response, parsed, dict(record), last_error)
            if retry_policy is not None:
                if retry_policy.retry(error_kind, last_error):
                    if error_kind == "format" and not getattr(retry_policy, "preserve_messages", False):
                        feedback = dict(validation_error=str(last_error),
                            instruction="Correct the response format using the original evidence and output contract. Return valid JSON.")
                        if parsed is not None:
                            feedback["previous_response"] = parsed
                        # Replace the latest format feedback instead of accumulating a conversation.
                        payload["messages"][1]["content"] = [*content,
                            {"type": "text", "text": json.dumps(feedback)}]
                    continue
                raise last_error
            if attempt >= self.max_attempts:
                break
            time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"VLM request failed after {self.max_attempts} attempts: {last_error}"
        ) from last_error

    def usage_report(self) -> dict[str, Any]:
        return self.usage.report()
