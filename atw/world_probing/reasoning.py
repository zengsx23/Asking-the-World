"""Collect engine evidence, then answer in an independent model conversation."""
import json
from pathlib import Path

from atw.world_modeling import sam3_protocol as s
from atw.openrouter import OpenRouterClient
from atw.world_probing.evidence import media, write
from atw.world_probing import final_answer
from atw.world_probing.collection_input import ToolFeedback, identity_bindings, handoff_instruction
from atw.world_probing.tool_definitions import tool

PROMPTS=Path(__file__).parents[1]/'prompts'
HANDOFF_SCHEMA=s.obj(status=dict(type='string',enum=['evidence_ready','evidence_unavailable']),reason=s.TEXT)


def has_engine_operation(trace, operation_names, *, successful=True):
    return any(r['tool_name'] in operation_names and (not successful or r['status']=='ok') for r in trace)


def finish_tool(trace, operation_names):
    def execute(arguments,context):
        if not has_engine_operation(trace,operation_names):
            raise ValueError('Collect engine evidence before finishing')
        return dict(arguments),[]
    return tool('finish_evidence_collection','Finish collection for final answering, or report missing evidence. Call alone.',
                HANDOFF_SCHEMA,HANDOFF_SCHEMA,execute)


def run(inputs, tools, question, mode, binding, output):
    output.mkdir(parents=True,exist_ok=False)
    artifacts=[]
    try:
        answer=collect(inputs,tools,question,mode,binding,output,artifacts)
    except (ValueError,KeyError,OSError,RuntimeError,TypeError) as exc:
        answer=unresolved(question,f'{type(exc).__name__}: {exc}')
    if answer['status']=='unresolved':
        answer=final_answer.from_video(tools.vlm,output/'video_fallback',question,
            inputs['benchmark'],inputs['video_path'],mode=mode,
            engine=inputs['stage_one_result']['data']['scene']['engine'],reason=answer['reason'])
    write(output/'result.json',answer)
    return answer,artifacts


def collect(inputs, tools, question, mode, binding, output, artifacts):
    engine=None;failure=None
    try:
        from atw.world_probing.engines import create
        engine=create(inputs,tools,question,mode,binding,output)
    except (ValueError,KeyError,OSError,RuntimeError,TypeError) as exc:
        failure=f'{type(exc).__name__}: {exc}'
    trace=[];failed_calls=set();arguments={};completed_calls={};reused_calls=[]
    definitions=engine.definitions() if engine is not None else []
    # World loading and control topology are setup, not outcome measurements.
    operation_names={d.DEFINITION['name'] for d in definitions}-{'load_baseline_world','query_rope_controls'}
    finish=finish_tool(trace,operation_names)
    tools.register_question_tools(question['id'],definitions)
    schemas=[d.DEFINITION for d in definitions]
    prompt_name=getattr(engine,'prompt_name',f'stage_two_{mode}.txt')
    prompt=(PROMPTS/'stage_two_reasoning.txt').read_text()+'\n'+(PROMPTS/prompt_name).read_text()
    payload=dict(question={k:question[k] for k in ('id','text','options','answer_format','answer_instructions') if k in question},
        identity_evidence=identity_bindings(binding['identity_evidence']),engine=engine.context() if engine else dict(unavailable=failure))
    write(output/'input.json',dict(system=prompt,context=payload,tools=schemas,handoff_schema=HANDOFF_SCHEMA))
    messages=[dict(role='system',content=prompt),
              dict(role='user',content=json.dumps(payload))]
    feedback=ToolFeedback()
    def validate(value):
        if 'tool_calls' in value:
            calls=value['tool_calls']
            if any(c['tool_name']=='finish_evidence_collection' for c in calls) and len(calls)!=1:
                raise ValueError('Finish collection alone')
            if any(c['tool_name']=='finish_evidence_collection' for c in calls) and not has_engine_operation(trace,operation_names):
                raise ValueError('Finish collection is not available')
            return value
        s.validate(HANDOFF_SCHEMA,value,'evidence handoff')
        if value['status']=='evidence_ready' and not has_engine_operation(trace,operation_names):
            raise ValueError('Collect engine evidence before handing off for a physical answer')
        if value['status']=='evidence_unavailable' and not has_engine_operation(trace,operation_names,successful=False):
            raise ValueError('Attempt an engine operation before reporting unavailable evidence')
        return value
    try:
        turn=0
        handoff=(None if operation_names else dict(status='evidence_unavailable',
            reason=failure or 'No engine query or simulation supports this question type.'))
        while True:
            if handoff is None:
                turn+=1
                request_dir=output/f'turn_{turn:04d}'
                ready=has_engine_operation(trace,operation_names)
                available=definitions+([finish] if ready else [])
                tools.register_question_tools(question['id'],available)
                schemas=[d.DEFINITION for d in available]
                messages[0]=dict(role='system',content=prompt+('\n'+handoff_instruction(HANDOFF_SCHEMA) if ready else ''))
                result=OpenRouterClient(tools.vlm).complete(messages,request_dir,validate=validate,tool_definitions=schemas)
                write(request_dir/'tools.json',schemas)
            else:result=handoff
            if 'tool_calls' not in result:
                write(output/'handoff.json',result)
                if result['status']=='evidence_ready':
                    if not has_engine_operation(trace,operation_names):
                        raise ValueError('Cannot answer before an engine query or simulation completes')
                    evidence,paths=final_answer.collect(trace,arguments,tools.run_dir)
                    from atw.world_probing.prediction_scope import applies
                    scope = (dict(observed_end_seconds=engine.backend.end)
                        if applies(mode, inputs['stage_one_result']['data']['scene']['engine']) else {})
                    answer=final_answer.run(tools.vlm,output/'final_answer',question,inputs['benchmark'],evidence,
                        binding=binding['identity_evidence'],paths=paths,mode=mode,
                        engine=inputs['stage_one_result']['data']['scene']['engine'], **scope)
                else:answer=unresolved(question,result['reason'])
                break
            provider=json.loads((request_dir/'provider_response.json').read_text())['choices'][0]['message']
            messages.append(dict(role='assistant',content=None,tool_calls=provider['tool_calls']))
            provider_ids=json.loads((request_dir/'provider_call_ids.json').read_text())
            observation_media=[];executed=0
            for call,pid in zip(result['tool_calls'],provider_ids):
                signature=json.dumps(call,sort_keys=True)
                if signature in failed_calls:
                    handoff=dict(status='evidence_unavailable',
                        reason='An unchanged failed tool call was repeated; the requested evidence could not be collected.')
                    break
                if signature in completed_calls:
                    receipt=completed_calls[signature]
                    visible=dict(call_id=receipt['call_id'],status='ok',reused_result=True)
                    messages.append(dict(role='tool',tool_call_id=pid,content=json.dumps(visible)))
                    reused_calls.append(dict(provider_call_id=pid,tool_name=call['tool_name'],arguments=call['arguments'],
                                             reused_call_id=receipt['call_id']))
                    write(output/'reused_calls.json',reused_calls)
                    continue
                receipt=tools.call(call['tool_name'],call['arguments'],stage='stage_two',question=question)
                executed+=1
                arguments[receipt['call_id']]=call['arguments']
                trace.append(receipt);artifacts.extend(receipt['artifacts'])
                if receipt['status']=='error':failed_calls.add(signature)
                else:completed_calls[signature]=receipt
                visible=feedback.view(receipt)
                write(output/'feedback'/f"{receipt['call_id']}.json",visible)
                messages.append(dict(role='tool',tool_call_id=pid,content=json.dumps(visible)))
                images=[a for a in receipt['artifacts'] if a['media_type'].startswith(('image/','video/'))]
                for item in images:
                    observation_media.extend([dict(type='text',text=f"Observed evidence: {item['artifact_id']} ({item['source_kind']})."),
                                              media(tools.run_dir/item['path'])])
                if call['tool_name']=='finish_evidence_collection' and receipt['status']=='ok':
                    handoff=receipt['data']
            if observation_media:messages.append(dict(role='user',content=observation_media))
            write(output/'trace.json',trace)
            if handoff is None and executed==0 and has_engine_operation(trace,operation_names):
                handoff=dict(status='evidence_ready',reason='All requested operations already have retained successful results; the final model must assess the available evidence.')
        return answer
    finally:
        write(output/'trace.json',trace)
        if engine is not None:engine.backend.close()


def unresolved(question,reason):
    return dict(question_id=question['id'],status='unresolved',answer=None,evidence_ids=[],reason=reason)
