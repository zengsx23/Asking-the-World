"""Question-only classification with the agent request policy."""
import json

from atw.shared.question_type_classifier import SYSTEM, validate_classification
from atw.openrouter import OpenRouterClient


def classify(question, output_dir, config):
    """Only literal text and option texts enter the request; metadata stays local."""
    payload = dict(question=question['text'],
                   options=[option['text'] for option in question['options']])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'input.json').write_text(json.dumps(dict(system=SYSTEM, context=payload), indent=2) + '\n')
    return OpenRouterClient(config).complete([
        dict(role='system', content=SYSTEM),
        dict(role='user', content=json.dumps(payload)),
    ], output_dir / 'request', validate=validate_classification)
