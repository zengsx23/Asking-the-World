"""One visual checkpoint on a frozen inventory before SAM3 propagation."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from atw.world_modeling import sam3_protocol as c
from atw.openrouter import OpenRouterClient
from atw.world_modeling.scene_assets import write

PROMPT = Path(__file__).resolve().parents[1]/'prompts/sam3_seed_review.txt'


def schema(ids):
    return c.obj(resegment=c.array(c.obj(object_id=dict(type='string',enum=ids),
        reason=c.TEXT)))


def validate(value, ids):
    c.validate(schema(ids), value, 'seed_review')
    c.unique_ids(value['resegment'], 'object_id', 'seed_review')
    return value


def render_pair(image, mask, label, frame):
    """Full context and proportional crop; only red contours alter RGB pixels."""
    if mask.shape != image.shape[:2] or not mask.any():
        raise ValueError('Review needs a nonempty seed mask matching the image')
    overlay=image.copy()
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,0,255),2,cv2.LINE_AA)
    y,x=np.nonzero(mask);pad=max(12,round(.2*max(np.ptp(x)+1,np.ptp(y)+1)))
    crop=overlay[max(0,y.min()-pad):min(image.shape[0],y.max()+pad+1),
                 max(0,x.min()-pad):min(image.shape[1],x.max()+pad+1)]
    panels=[]
    for source,title in ((overlay,'context'),(crop,'crop')):
        canvas=np.full((464,640,3),255,np.uint8)
        scale=min(640/source.shape[1],420/source.shape[0])
        view=cv2.resize(source,(max(1,round(source.shape[1]*scale)),max(1,round(source.shape[0]*scale))))
        x0=(640-view.shape[1])//2;y0=44+(420-view.shape[0])//2
        canvas[y0:y0+view.shape[0],x0:x0+view.shape[1]]=view
        cv2.putText(canvas,f'{label} | frame {frame} | {title}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
        panels.append(canvas)
    return np.hstack(panels)


def review(*, objects, seeds, video_path, output, vlm, prompt=None):
    """Only inventory descriptions and labeled images enter a fresh request."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ids=[o['target_id'] for o in objects]
    if set(ids)!=set(seeds):raise ValueError('Review inventory and seed identities differ')
    inventory=[dict(object_id=o['target_id'],description=o['discriminative_description']) for o in objects]
    write(output/'inventory.json',dict(objects=inventory))
    if not objects:
        decision=dict(resegment=[]);write(output/'decision.json',decision);return decision
    capture=cv2.VideoCapture(str(video_path));images=[]
    try:
        for index,obj in enumerate(objects):
            key=obj['target_id'];seed=seeds[key];frame=int(seed['frame_index'])
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
            if not ok:raise ValueError(f'Cannot decode seed frame {frame}')
            pair=render_pair(image,np.asarray(seed['mask'],dtype=bool),f'O{index+1}',frame)
            path=output/f'object_{index+1:03d}.png'
            if not cv2.imwrite(str(path),pair):raise OSError(path)
            inventory[index]['image_label']=f'O{index+1}'
            media='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
            images.append(dict(type='image_url',image_url=dict(url=media)))
    finally:capture.release()
    content=[dict(type='text',text=json.dumps(dict(objects=inventory),ensure_ascii=False)+'\n'+c.prompt(schema(ids))),*images]
    decision=OpenRouterClient(vlm).complete([
        dict(role='system',content=PROMPT.read_text() if prompt is None else prompt),
        dict(role='user',content=content)],output/'request',validate=lambda v:validate(v,ids))
    write(output/'inventory.json',dict(objects=inventory));write(output/'decision.json',decision)
    return decision


def duplicate_candidates(target_id, frame, mask, seeds):
    from atw.perception.sam3_tool import _mask_iou, SEED_DUPLICATE_IOU
    return [key for key,seed in seeds.items() if key!=target_id and seed['frame_index']==frame
            and _mask_iou(mask,seed['mask'])>=SEED_DUPLICATE_IOU]


def repair(*, decision, objects, seeds, manifest, video_path, output, call_prefix, runtime, agent):
    """Reselect each flagged identity once; never alter the locked inventory."""
    from atw.perception import sam3_tool as sam
    from atw.world_modeling.sam3_seeds import select_primary_masks
    from atw.world_modeling.sam3_containment import contained_objects
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    validate(decision,[o['target_id'] for o in objects])
    current=dict(seeds);records=list(manifest);by_id={o['target_id']:o for o in objects}
    pending={item['object_id'] for item in decision['resegment']}
    # Rejected assignments no longer reserve an identity. Otherwise a swapped
    # pair could never select its correct masks. Accepted seeds stay reserved.
    reserved={key:seed for key,seed in seeds.items() if key not in pending}
    count,*_=sam._video_metadata(video_path);sampled=sam._sampled_frame_indices(count,maximum=8)
    for item in decision['resegment']:
        key=item['object_id'];directory=output/key;directory.mkdir(parents=True,exist_ok=False)
        record=next(r for r in records if r['target']['target_id']==key)
        target={**deepcopy(record['target']),**deepcopy(by_id[key])}
        # Remove prior decision context; identity and query phrases remain fixed.
        for field in ('seed_containment_feedback','seed_containment_evidence'):
            target.pop(field,None)
        target['primary_seed_frame']=current[key]['frame_index']
        target['seed_review_feedback']=item['reason']+' Keep the fixed inventory identity and description.'
        target['primary_seed_reason']=''
        sheet=directory/'frames.jpg';sam._write_indexed_contact_sheet(video_path,sampled,sheet)
        timings={}
        replacement,new_records,arrays=select_primary_masks(agent=agent,runtime=runtime,
            video_path=video_path,request_dir=directory,call_prefix=call_prefix+'_'+key,
            objects=[target],timings=timings,contact_sheet_path=sheet,sampled_frames=sampled,
            fixed_identity=True,
            candidate_exclusion=lambda k,f,m:duplicate_candidates(k,f,m,reserved),
            candidate_filter=lambda k,f,m:contained_objects(k,f,m,reserved))
        write(directory/'timings.json',timings);write(directory/'seed_manifest.json',new_records)
        np.savez_compressed(directory/'selected_primary_masks.npz',**arrays)
        if key not in replacement:
            fallback = dict(action='retain_initial_seed',
                reason='Independent reselection exhausted without an acceptable mask',
                review_reason=item['reason'], frame_index=int(seeds[key]['frame_index']))
            write(directory/'fallback.json', fallback)
            retained = deepcopy(record)
            retained['visual_review_fallback'] = fallback
            records = [retained if r['target']['target_id'] == key else r for r in records]
            reserved[key] = seeds[key]
            continue
        current[key]=replacement[key]
        reserved[key]=replacement[key]
        records=[r for r in records if r['target']['target_id']!=key]+new_records
    return current,records
