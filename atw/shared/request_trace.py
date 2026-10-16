"""Opt-in, ordered model/tool I/O export without headers or credentials."""
import base64
import copy
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stage(root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ordinal = len(list(root.iterdir())) + 1
    path = root / f"{ordinal:03d}_{name}"
    (path / "input").mkdir(parents=True)
    (path / "output").mkdir()
    return path


def request_trace(root, operation, call_id, attempt, payload):
    path = stage(root, f"{operation}_{call_id}_attempt{attempt}")
    request = copy.deepcopy(payload)
    for message_index, message in enumerate(request["messages"]):
        if not isinstance(message.get("content"), list):
            continue
        for index, part in enumerate(message.get("content")):
            kind = part.get("type")
            if kind not in {"image_url", "video_url"}:
                continue
            url = part[kind]["url"]
            if not url.startswith("data:"):
                continue
            header, encoded = url.split(",", 1)
            mime = header[5:].split(";")[0]
            suffix = {"image/jpeg":".jpg", "image/png":".png", "video/mp4":".mp4"}.get(mime, ".bin")
            filename = f"media_{message_index:02d}_{index:02d}{suffix}"
            (path / "input" / filename).write_bytes(base64.b64decode(encoded))
            part[kind]["url"] = filename
            part["export_mime_type"] = mime
    write(path / "input/request.json", request)
    text_messages = []
    for message_index, message in enumerate(request["messages"]):
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") == "text":
                    try:
                        texts.append(json.loads(part["text"]))
                    except json.JSONDecodeError:
                        texts.append(part["text"])
            content = texts
        text_messages.append({**message, "content": content})
    write(path / "input/text.json", text_messages)
    if "tools" in request:
        write(path / "input/tools.json", request["tools"])
    return path


def response_trace(path, raw, parsed, usage, error):
    write(path / "output/response.json", dict(raw_response=raw, parsed_response=parsed,
                                             usage=usage, error_type=type(error).__name__ if error else None))


def tool_trace(root, question_id, evidence_id, call, receipt):
    path = stage(root, f"tool_q{question_id}_{evidence_id}_{call['tool']}")
    write(path / "input/request.json", call)
    write(path / "output/response.json", receipt)
