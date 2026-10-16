"""Full-frame and crop candidate association and visual evidence."""
import cv2
import numpy as np

def retain_local_candidates(candidates,proposals,shape,per_view=2):
    h,w=shape;targets=np.asarray([[p['box']['left']*w/1000,p['box']['top']*h/1000,p['box']['right']*w/1000,p['box']['bottom']*h/1000] for p in proposals]);areas=(targets[:,2]-targets[:,0])*(targets[:,3]-targets[:,1]);groups={}
    for candidate in candidates:
        yy,xx=np.where(candidate['mask'])
        if not len(xx):continue
        extent=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        intersection=np.maximum(0,np.minimum(targets[:,2],extent[2])-np.maximum(targets[:,0],extent[0]))*np.maximum(0,np.minimum(targets[:,3],extent[3])-np.maximum(targets[:,1],extent[1]))
        scores=intersection/np.maximum(1,areas+(extent[2]-extent[0])*(extent[3]-extent[1])-intersection);owner=int(scores.argmax())
        if scores[owner]==0:continue
        mode='FULL' if '_FULL' in candidate['candidate_id'] else 'CROP'
        item=dict(candidate,generated_for=candidate['object_id'],object_id=proposals[owner]['id'],box_iou=float(scores[owner]))
        groups.setdefault((owner,mode),[]).append(item)
    retained=[]
    for key in sorted(groups):
        chosen=[]
        for item in sorted(groups[key],key=lambda c:(c['box_iou'],c['score']),reverse=True):
            if any(np.array_equal(item['mask'],other['mask']) for other in chosen):continue
            chosen.append(item)
            if len(chosen)==per_view:break
        retained.extend(chosen)
    return [dict(item,source_candidate_id=item['candidate_id'],candidate_id=f'C{i:03d}') for i,item in enumerate(retained,1)]

def boards(raw,proposals,candidates,out):
    h,w=raw.shape[:2];paths=[]
    for proposal in proposals:
        b=proposal['box'];x,y,r,bot=b['left']*w/1000,b['top']*h/1000,b['right']*w/1000,b['bottom']*h/1000
        pad=round(max(r-x,bot-y)*.25)+12;x,y,r,bot=max(0,int(x-pad)),max(0,int(y-pad)),min(w,int(r+pad)),min(h,int(bot+pad))
        choices=[c for c in candidates if c['object_id']==proposal['id']]
        panels=[('ORIGINAL',raw[y:bot,x:r],None)]+[(c['candidate_id'],cv2.cvtColor(c['mask'][y:bot,x:r].astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),c['mask']) for c in choices]
        tiles=[]
        for title,panel,mask in panels:
            tile=np.full((420,270,3),30,np.uint8);scale=min(250/panel.shape[1],240/panel.shape[0]);im=cv2.resize(panel,(max(1,round(panel.shape[1]*scale)),max(1,round(panel.shape[0]*scale))),interpolation=cv2.INTER_NEAREST);tile[35:35+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(tile,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            mini=raw if mask is None else cv2.cvtColor(mask.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)
            mini=cv2.resize(mini,(250,141),interpolation=cv2.INTER_NEAREST);cv2.rectangle(mini,(round(x*250/w),round(y*141/h)),(round(r*250/w),round(bot*141/h)),(0,0,255),1);tile[277:418,10:260]=mini;tiles.append(tile)
        strip=np.hstack(tiles);canvas=np.full((strip.shape[0]+35,strip.shape[1],3),30,np.uint8);canvas[35:]=strip
        cv2.putText(canvas,proposal['id']+' | '+proposal['name'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        path=out/f"{proposal['id']}_local_board.jpg";cv2.imwrite(str(path),canvas);paths.append(path)
    return paths


def generate(raw, proposals, domain, runtime):
    """Two native SAM3 views per proposal; retain visible material inside the enclosure."""
    import torch
    from PIL import Image
    h, w = raw.shape[:2]
    processor = runtime.image_processor
    candidates = []
    with runtime.inference_lock, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full_state = processor.set_image(Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)))
        for proposal in proposals:
            box = proposal['box']
            x, y, r, b = box['left']*w/1000, box['top']*h/1000, box['right']*w/1000, box['bottom']*h/1000
            for mode in ('FULL', 'CROP'):
                if mode == 'FULL':
                    a, t, c, d = 0, 0, w, h
                    state = dict(full_state)
                    state['backbone_out'] = dict(full_state['backbone_out'])
                else:
                    pad = round(max(r-x, b-y)*.2)+8
                    a, t, c, d = max(0, int(x-pad)), max(0, int(y-pad)), min(w, int(r+pad)), min(h, int(b+pad))
                    state = processor.set_image(Image.fromarray(cv2.cvtColor(raw[t:d, a:c], cv2.COLOR_BGR2RGB)))
                processor.reset_all_prompts(state)
                state['backbone_out'].update(processor.model.backbone.forward_text([proposal['name']], device=processor.device))
                response = processor.add_geometric_prompt(
                    [((x+r)/2-a)/(c-a), ((y+b)/2-t)/(d-t), (r-x)/(c-a), (b-y)/(d-t)], True, state)
                scores = response['scores'].detach().float().cpu().numpy().reshape(-1)
                masks = response['masks'].detach().cpu().numpy().reshape(-1, d-t, c-a)
                for j, (mask, score) in enumerate(zip(masks, scores)):
                    full = np.zeros((h, w), bool)
                    full[t:d, a:c] = mask
                    full &= domain
                    if full.any():
                        candidates.append(dict(object_id=proposal['id'], candidate_id=f"{proposal['id']}_{mode}{j}",
                                               score=float(score), mask=full))
    return candidates


def compose(raw, candidates, decision):
    """Candidate unions and highest-SAM-score ownership; no post-hoc object merging."""
    h, w = raw.shape[:2]
    lookup = {c['candidate_id']: c for c in candidates}
    labels = np.zeros((h, w), np.uint16)
    best = np.full((h, w), -np.inf)
    rows = []
    for i, row in enumerate(decision['objects'], 1):
        mask = np.logical_or.reduce([lookup[cid]['mask'] for cid in row['candidate_ids']])
        score = max(lookup[cid]['score'] for cid in row['candidate_ids'])
        take = mask & (score > best)
        labels[take], best[take] = i, score
        yy, xx = np.where(mask)
        rows.append(dict(id=f'S{i}', name=row['name'], description=row['name'],
            box=dict(top=float(yy.min()*1000/h), left=float(xx.min()*1000/w),
                     bottom=float((yy.max()+1)*1000/h), right=float((xx.max()+1)*1000/w)),
            candidate_ids=row['candidate_ids']))
        if 'is_receiving_structure' in row:
            rows[-1]['is_receiving_structure'] = row['is_receiving_structure']
    return labels, rows, render_overview(raw, labels, rows)


def render_overview(raw, labels, rows):
    """Show the final static identities after candidate selection/refinement."""
    overview = raw.copy()
    for i, row in enumerate(rows, 1):
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overview, contours, -1, (0, 255, 255), 1)
        yy, xx = np.where(mask)
        if len(xx):
            cv2.putText(overview, row['id'], (int(xx.min()), max(16, int(yy.min())-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return overview
