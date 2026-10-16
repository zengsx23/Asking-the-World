#!/usr/bin/env python3
"""Score saved answers offline; CLEVRER and ContPhy are reported separately."""
import argparse
import collections
import json
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def normalize(value,options):
    if value is None:return None
    if options:
        values=value if isinstance(value,list) else [value]
        values=[str(v) for v in values]
        if len(values)!=len(set(values)) or set(values)-set(options):return None
        return sorted(values)
    if isinstance(value,bool):return 'yes' if value else 'no'
    if isinstance(value,(int,float)) and float(value).is_integer():return str(int(value))
    value=' '.join(str(value).lower().strip().split()).rstrip('.')
    return {'cannot answer':'can not answer','can\'t answer':'can not answer'}.get(value,value)

def summarize(rows):
    n=len(rows); answered=sum(r['answered'] for r in rows); correct=sum(r['correct'] for r in rows)
    opts=sum(r['option_total'] for r in rows); answered_opts=sum(r['option_total'] for r in rows if r['answered'])
    option_correct=sum(r['option_correct'] for r in rows)
    scenes=collections.defaultdict(list)
    for r in rows:scenes[r['scene']].append(r)
    finished=[v for v in scenes.values() if all(r['answered'] for r in v)]
    perfect=sum(all(r['correct'] for r in v) for v in finished)
    ratio=lambda a,b:a/b if b else None
    fallback=[r for r in rows if r['fallback_triggered']]
    return dict(questions=n,answered=answered,unanswered=n-answered,correct=correct,
        coverage=ratio(answered,n),accuracy_answered=ratio(correct,answered),accuracy_all=ratio(correct,n),
        option_total=opts,option_answered=answered_opts,option_correct=option_correct,
        option_accuracy_answered=ratio(option_correct,answered_opts),option_accuracy_all=ratio(option_correct,opts),
        scenes=len(scenes),fully_answered_scenes=len(finished),perfect_scenes=perfect,
        scene_accuracy_completed=ratio(perfect,len(finished)),scene_accuracy_all=ratio(perfect,len(scenes)),
        fallback_triggered=len(fallback),fallback_answered=sum(r['answered'] for r in fallback),
        fallback_correct=sum(r['correct'] for r in fallback),
        fallback_accuracy_answered=ratio(sum(r['correct'] for r in fallback),sum(r['answered'] for r in fallback)))


def evaluate(reference, manifest):
    reference, manifest = Path(reference), Path(manifest).resolve()
    refs = read(reference)['questions']
    runs = read(manifest)
    if not isinstance(runs, dict):
        raise ValueError('Run manifest must map scene IDs to run directories')
    known = {r['scene'] for r in refs}
    if set(runs) - known:
        raise ValueError(f'Unknown scenes in manifest: {sorted(set(runs)-known)}')
    selected = [r for r in refs if r['scene'] in runs]
    cache, indices, rows = {}, {}, []
    seen = set()
    for r in selected:
        sid = r['scene']
        key = (sid, r['id'])
        if key in seen:
            raise ValueError(f'Duplicate reference: {key}')
        seen.add(key)
        run = Path(runs[sid])
        if not run.is_absolute():
            run = manifest.parent/run
        if sid not in cache:
            answers = read(run/'stage_two/answers.json', {}).get('answers', [])
            if not answers:
                answers = read(run/'stage_two/result.json', {}).get('data', {}).get('answers', [])
            cache[sid] = {a['question_id']: a for a in answers}
            if len(cache[sid]) != len(answers):
                raise ValueError(f'Duplicate answers: {sid}')
            # Fallback artifacts use the question order in the actual request.
            request = read(run/'input.json')
            questions = request['questions'] if request else [x['question'] for x in selected if x['scene'] == sid]
            questions = [q for q in questions if q.get('question_type') != 'descriptive']
            indices[sid] = {q['id']: i for i, q in enumerate(questions)}
        answer = cache[sid].get(r['id'])
        options = [o['id'] for o in r['question']['options']]
        expected = normalize(r['expected'], options)
        if expected is None:
            raise ValueError(f'Invalid reference answer: {key}')
        prediction = normalize(answer.get('answer'), options) if answer else None
        answered = bool(answer and answer.get('status') == 'answered' and prediction is not None)
        index = indices[sid].get(r['id'])
        trigger = read(run/'stage_two/reasoning'/f'q_{index:04d}'/'video_fallback_trigger.json') if index is not None else None
        fallback = bool(trigger or (answer and 'input_video' in answer.get('evidence_ids', [])))
        rows.append(dict(scene=sid, group=r['group'], id=r['id'], qtype=r['qtype'], family=r['family'],
            expected=expected, predicted=prediction, answered=answered, correct=answered and prediction == expected,
            option_total=len(options), option_correct=sum((o in prediction) == (o in expected) for o in options) if answered else 0,
            fallback_triggered=fallback, fallback_reason=(trigger or {}).get('reason')))
    def belongs(r, group):
        return r['group'] != 'clevrer' if group == 'contphy' else r['group'] == group
    groups = ('clevrer', 'contphy', 'cloth', 'softbody', 'rope', 'fluid')
    result = dict(groups={g: summarize([r for r in rows if belongs(r, g)]) for g in groups},
        by_question_type={g: {t: summarize([r for r in rows if belongs(r,g) and r['qtype']==t])
                             for t in sorted({r['qtype'] for r in rows if belongs(r,g)})} for g in groups},
        scenes={sid: summarize([r for r in rows if r['scene']==sid]) for sid in sorted(runs)}, rows=rows)
    def category(qtype):
        for marker, label in (('counterfactual','counterfactual'), ('predictive','predictive'), ('goaldriven','goal')):
            if marker in qtype:
                return label
        return 'property'
    result['contphy_categories'] = {
        g: {t: summarize([r for r in rows if belongs(r,g) and category(r['qtype'])==t])
            for t in ('property', 'predictive', 'counterfactual', 'goal')}
        for g in groups if g != 'clevrer'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='JSON mapping scene IDs to run directories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.reference, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({g: result['groups'][g] for g in ('clevrer','contphy')}, indent=2))
