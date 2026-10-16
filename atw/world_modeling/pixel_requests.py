"""Native Agentic requests; all failures share the independent client's retry budget."""
import base64
import mimetypes
from pathlib import Path

from atw.openrouter import OpenRouterClient
from atw.world_modeling import sam3_protocol as schemas
from atw.world_modeling.scene_assets import write

SYSTEM = 'Inspect observed images and return exactly one JSON object satisfying the supplied schema. No Markdown.'


def prompt(name):
    return (Path(__file__).resolve().parents[1] / 'prompts' / f'pixel_{name}.txt').read_text().strip()


class PixelRequests:
    def __init__(self, context):
        self.context = context
        self.client = OpenRouterClient(context.vlm)

    def call(self, operation, instruction, schema, media, check=None, *, system=SYSTEM):
        output = self.context.output_dir / 'requests' / operation
        instruction += schemas.prompt(schema)
        write(output / 'request.json', dict(system=system, prompt=instruction,
              media=[str(p) for p in media], output_schema=schema))
        content = [dict(type='text', text=instruction)]
        for path in media:
            mime = mimetypes.guess_type(str(path))[0]
            if mime is None or mime.split('/')[0] not in ('image', 'video'):
                raise ValueError(f'Unsupported visual input: {path}')
            kind = 'video_url' if mime.startswith('video/') else 'image_url'
            url = f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')
            content.append(dict(type=kind, **{kind: dict(url=url)}))

        def validate(value):
            schemas.validate(schema, value)
            if check is not None:
                check(value)
            return value

        return self.client.complete([dict(role='system', content=system),
                                     dict(role='user', content=content)], output / 'response', validate=validate)
