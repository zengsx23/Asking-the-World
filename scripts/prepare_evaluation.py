#!/usr/bin/env python3
"""Create benchmark requests and separate, offline-only answer references."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

CONT = {'cloth': ('cloth/cloth', 'cloth_final2'),
        'softbody': ('ball/soft', 'soft_final'),
        'rope': ('rope/rope', 'rope_final2'),
        'fluid': ('fluid/fluid', 'fluid_final')}


def prepare(data_root, output):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    requests, references, hashes = {}, [], {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(data_root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def add(group, number, video, question, expected, qtype, family):
        sid = f'{group}_{number}'
        request = requests.setdefault(sid, dict(protocol_version=1,
            video_path=str(video), benchmark='clevrer' if group == 'clevrer' else 'contphy', questions=[]))
        if any(q['id'] == question['id'] for q in request['questions']):
            raise ValueError(f'Duplicate question: {sid}/{question["id"]}')
        request['questions'].append(question)
        references.append(dict(scene=sid, group=group, id=question['id'], question=question,
                               expected=expected, qtype=qtype, family=family))

    for scene in read(data_root/'clevrer/validation.json'):
        number = int(scene['scene_index'])
        if not 10000 <= number <= 10999:
            continue
        for q in scene['questions']:
            if q['question_type'] == 'descriptive':
                continue
            question = dict(id=str(q['question_id']), text=q['question'],
                options=[dict(id=str(c['choice_id']), text=c['choice']) for c in q['choices']],
                answer_format='multi_select', question_type=q['question_type'])
            add('clevrer', number, data_root/f'clevrer/video_10000-11000/video_{number}.mp4',
                question, [str(c['choice_id']) for c in q['choices'] if c['answer'] == 'correct'],
                q['question_type'], q.get('question_subtype', q['question_type']))
    for group, (folder, prefix) in CONT.items():
        base = data_root/'contphy/full'/folder
        for q in read(base/'question'/f'{prefix}_test_v1.json'):
            number = int(q['video_id'])
            if not 350 <= number <= 499:
                continue
            choices = re.findall(r'(?:^|\n)([A-Z])\.\s*([^\n]+)', q['question'])
            question = dict(id=str(q['question_id']), text=re.split(r'\nA\.\s*', q['question'])[0].strip(),
                options=[dict(id=a, text=b.strip()) for a, b in choices],
                answer_format='multi_select' if choices else 'open_ended')
            add(group, number, base/'video'/str(number)/'output_Full.mp4', question,
                re.findall(r'[A-Z]', q['answer']) if choices else q['answer'], q['qtype'], q['question_family'])
    expected_scenes = {f'clevrer_{i}' for i in range(10000, 11000)}
    expected_scenes.update(f'{g}_{i}' for g in CONT for i in range(350, 500))
    if set(requests) != expected_scenes:
        raise ValueError(f'Incomplete subset: missing {sorted(expected_scenes-set(requests))}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    for sid, request in requests.items():
        (output/'inputs'/f'{sid}.json').write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n')
    (output/'reference.json').write_text(json.dumps(dict(source_sha256=hashes, questions=references),
                                                   ensure_ascii=False, indent=2)+'\n')
    (output/'runs.json').write_text(json.dumps({sid: f'runs/{sid}' for sid in sorted(requests)}, indent=2)+'\n')
    return dict(scenes=len(requests), questions=dict(collections.Counter(r['group'] for r in references)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output_dir), indent=2))
