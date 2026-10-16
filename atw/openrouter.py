"""Independent JSON requests with validation, bounded retries and raw receipts."""
from __future__ import annotations

from http.client import HTTPException
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from atw.config import VLMConfig
from atw.request_policy import MAX_RETRIES, RequestRetries

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class MissingCredentialError(RuntimeError):
    pass


class RequestError(RuntimeError):
    def __init__(self, message: str, response: dict | None = None):
        super().__init__(message)
        self.response = response


class OpenRouterClient:
    def __init__(self, config: VLMConfig):
        self.config = config

    def complete(
        self, messages: list[dict], output_dir: Path, *,
        validate: Callable[[dict], dict] | None = None,
        tool_definitions: list[dict] | None = None,
        normalize_response: Callable | None = None,
    ) -> dict:
        """Return parsed JSON, optionally validated by the caller's contract.

        Use a new output directory per invocation. Transport, parsing and
        validation share three retries, with unchanged messages each time.
        Validators return the accepted record or raise ValueError.
        """
        output = Path(output_dir)
        output.mkdir(parents=True)
        started = perf_counter()
        attempts = []
        retries = RequestRetries(sleep)
        for number in range(1, MAX_RETRIES + 2):
            attempt = output / "attempts" / f"{number:04d}"
            attempt.mkdir(parents=True)
            attempt_started = perf_counter()
            response = None
            failure = None
            try:
                response = (self._complete_once(messages, attempt, tool_definitions=tool_definitions)
                            if tool_definitions else self._complete_once(messages, attempt))
                message = response["choices"][0]["message"]
                content = message.get("content")
                native_calls = message.get('tool_calls') if tool_definitions else None
                if native_calls:
                    definitions = {d['name']:d for d in tool_definitions}
                    calls = []
                    provider_ids = []
                    from atw.world_modeling.sam3_protocol import validate as validate_schema
                    for native in native_calls:
                        function = native['function']
                        name = function['name']
                        if name not in definitions:
                            raise ValueError('Model requested an unavailable tool')
                        arguments = json.loads(function['arguments'])
                        validate_schema(definitions[name]['input_schema'], arguments, 'arguments')
                        calls.append(dict(tool_name=name, arguments=arguments))
                        provider_ids.append(native['id'])
                    result = dict(tool_calls=calls)
                    (attempt/'provider_call_ids.json').write_text(json.dumps(provider_ids)+'\n')
                else:
                    if not isinstance(content, str):
                        raise ValueError("Expected a text response")
                    (attempt / "response.txt").write_bytes(content.encode("utf-8"))
                    result = json.loads(content)
                    if normalize_response is not None:
                        result = normalize_response(result)
                    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                        result = result[0]
                    if not isinstance(result, dict):
                        raise ValueError("Expected a JSON object")
                if validate is not None:
                    result = validate(result)
            except (RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
                failure = exc
                if isinstance(exc, RequestError):
                    response = exc.response

            record = {
                "attempt": number, "seconds": perf_counter() - attempt_started,
                "status": "failed" if failure is not None else "completed",
                "model": response.get("model") if response else None,
                "response_id": response.get("id") if response else None,
                "usage": response.get("usage") if response else None,
                "error": ({"type": type(failure).__name__, "message": str(failure)}
                          if failure is not None else None),
            }
            (attempt / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            attempts.append(record)
            # Totals include every available receipt, including invalid completions.
            usage = {}
            for item in attempts:
                if isinstance(item["usage"], dict):
                    for key, value in item["usage"].items():
                        if type(value) in (int, float):
                            usage[key] = usage.get(key, 0) + value
            (output / "metrics.json").write_text(json.dumps({
                "seconds": perf_counter() - started,
                "model": record["model"], "response_id": record["response_id"],
                "usage": usage, "attempt_count": len(attempts),
                "usage_receipts_complete": all(isinstance(item["usage"], dict) for item in attempts),
                "attempts": attempts,
            }, indent=2) + "\n")
            if failure is not None:
                if not retries.retry("request", failure):
                    raise RuntimeError(
                        f"OpenRouter request failed after {number} requests ({MAX_RETRIES} retries): {failure}"
                    ) from failure
                continue

            # Request-level response files always refer to the successful completion.
            for name in ("provider_response.json", "response.txt", "provider_call_ids.json"):
                source = attempt / name
                if source.exists():
                    (output / name).write_bytes(source.read_bytes())
            (output / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result

    def _complete_once(self, messages: list[dict], output_dir: Path, *, tool_definitions=None) -> dict:
        # Credentials come from the process environment, never a project file.
        key = os.environ.get("OPEN_ROUTER_KEY")
        if not key:
            raise MissingCredentialError("OPEN_ROUTER_KEY is required for OpenRouter requests")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if tool_definitions:
            payload['tools'] = [dict(type='function', function=dict(name=d['name'],
                description=d['description'], parameters=d['input_schema'])) for d in tool_definitions]
        request = Request(CHAT_URL, data=json.dumps(payload).encode("utf-8"),
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=240) as response:
                body = response.read()
        except HTTPError as exc:
            try:
                body = exc.read()
            except (OSError, HTTPException) as read_error:
                raise RequestError(f"OpenRouter HTTP {exc.code}; error body read failed") from read_error
            (output_dir / "provider_error.txt").write_bytes(body)
            raise RequestError(f"OpenRouter HTTP {exc.code}") from exc
        except (OSError, HTTPException) as exc:
            raise RequestError(f"OpenRouter transport failed: {exc}") from exc
        (output_dir / "provider_response.json").write_bytes(body)
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("Expected an OpenRouter response object")
        if parsed.get("error"):
            raise RequestError("OpenRouter returned an error; see provider_response.json", parsed)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RequestError("Expected nonempty OpenRouter choices", parsed)
        for choice in choices:
            if not isinstance(choice, dict):
                raise RequestError("Expected an OpenRouter choice object", parsed)
            if choice.get("error") or choice.get("finish_reason") in ("error", "length", "content_filter"):
                raise RequestError("OpenRouter returned an unsuccessful completion; see provider_response.json", parsed)
        return parsed
