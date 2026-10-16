import base64


import json


import mimetypes


from pathlib import Path


from atw.shared.vlm_planner import _read_env_value


from atw.shared.vlm_usage import OpenRouterJSONClient, default_openrouter_transport


from atw.perception.pipeline.recording import write_json


PAGE_SIZE = 6


def media_content(evidence_id, path):
    mime = mimetypes.guess_type(str(path))[0]
    if mime is None or not mime.startswith(("image/", "video/")):
        raise ValueError(f"Unsupported review media: {path}")
    kind = "video_url" if mime.startswith("video/") else "image_url"
    return [{"type": "text", "text": evidence_id},
            {"type": kind, kind: {"url": f"data:{mime};base64," +
                                  base64.b64encode(Path(path).read_bytes()).decode()}}]


class SceneReviewer:
    def __init__(self, client):
        self.client = client

    @classmethod
    def from_project(cls, root, output):
        root, output = Path(root), Path(output)
        config = json.loads((root / "configs/llm/openrouter_google_gemini_3_flash_preview.json").read_text())
        if config["provider"] != "openrouter":
            raise ValueError("Scene review requires the configured OpenRouter provider")
        attempt = 0

        def capture(url, headers, payload):
            nonlocal attempt
            attempt += 1
            # Save actual non-secret request settings; omitted settings stay omitted.
            write_json(output / f"request_{attempt:03d}_settings.json",
                       {k: v for k, v in payload.items() if k != "messages"})
            response = default_openrouter_transport(url, headers, payload)
            write_json(output / f"provider_{attempt:03d}.json", response)
            return response

        client = OpenRouterJSONClient(
            api_key=_read_env_value(root / ".env", "OPEN_ROUTER_KEY"),
            model=config["model"], max_output_tokens=config["max_output_tokens"],
            module_name="scene_harness", transport=capture,
        )
        return cls(client)
