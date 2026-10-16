"""Discrete local material candidates; no model-generated pixel coordinates."""
import cv2,numpy as np
from atw.world_modeling import pixel_protocol as p
from atw.world_modeling.scene_assets import write
from atw.world_modeling.pixel_requests import prompt

def ambiguous(labels,rows,details):
    centers={r['id']:np.array(r['lab_center']) for r in details if 'lab_center' in r}
    ids=list(centers)
    for i,first in enumerate(ids):
        expanded=cv2.dilate((labels==int(first[1:])).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        for second in ids[:i]:
            if np.linalg.norm(centers[first]-centers[second])<6 and np.any(expanded&(labels==int(second[1:]))):
                return True
    return False

def review(raw,labels,rows,domain,requests,output,first_frame):
    targets=[r for r in rows if r['is_receiving_structure']]
    union=np.isin(labels,[int(r['id'][1:]) for r in targets]);yy,xx=np.where(union)
    h,w=labels.shape;x,y,r,b=max(0,xx.min()-20),max(0,yy.min()-20),min(w,xx.max()+21),min(h,yy.max()+21)
    protected=(labels!=0)&~union;allowed=domain&~protected
    raw_crop=raw[y:b,x:r];original=output/'identity_crop.jpg';cv2.imwrite(str(original),raw_crop)
    lab=cv2.cvtColor(raw,cv2.COLOR_BGR2LAB).astype(np.float32);lab[...,0]/=2.55
    values=lab[y:b,x:r].reshape(-1,3)
    cv2.setRNGSeed(41)
    _,groups,centers=cv2.kmeans(values[::2],12,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,.1),3,cv2.KMEANS_PP_CENTERS)
    distances=np.linalg.norm(values[:,None]-centers[None],axis=-1)
    classes=distances.argmin(-1).reshape(raw_crop.shape[:2])
    candidates=[dict(mask=labels==int(row['id'][1:]),origin='native:'+row['id']) for row in targets]
    for c in range(len(centers)):
        region=(classes==c)&allowed[y:b,x:r]
        if (region&union[y:b,x:r]).sum()<20:continue
        mask=np.zeros_like(union);mask[y:b,x:r]=region
        candidates.append(dict(mask=mask,origin=f'material_{c}'))
    pages=[]
    for i,c in enumerate(candidates,1):c['candidate_id']=f'C{i:03d}'
    np.savez_compressed(output/'identity_candidates.npz',**{c['candidate_id']:c['mask'] for c in candidates})
    write(output/'identity_candidates.json',[{k:v for k,v in c.items() if k!='mask'} for c in candidates])
    for start in range(0,len(candidates),4):
        tiles=[]
        for c in candidates[start:start+4]:
            selected=c['mask'][y:b,x:r];im=np.full_like(raw_crop,235);im[selected]=raw_crop[selected]
            scale=min(680/im.shape[1],300/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
            tile=np.full((340,700,3),235,np.uint8);tile[35:35+im.shape[0],10:10+im.shape[1]]=im
            cv2.putText(tile,c['candidate_id'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1);tiles.append(tile)
        while len(tiles)<4:tiles.append(np.full_like(tiles[0],235))
        path=output/f'identity_candidates_{start//4}.jpg';cv2.imwrite(str(path),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
        pages.append(path)
    def check(value):
        p.check_selection(value)
        if not value['objects']:raise ValueError('Return supported receiving structures')
    value=requests.call('container_identity_review',
        prompt('container_selection'),
        p.selection([c['candidate_id'] for c in candidates]),[original,*pages],check)
    lookup={c['candidate_id']:c['mask'] for c in candidates}
    objects=[r for r in rows if not r['is_receiving_structure']]
    result=labels.copy();result[union]=0
    index=max(int(row['id'][1:]) for row in rows)+1
    for row in value['objects']:
        mask=np.logical_or.reduce([lookup[k] for k in row['candidate_ids']])&allowed&(result==0)
        yy,xx=np.where(mask)
        if not len(xx):raise ValueError('Reviewed material selection is empty')
        result[mask]=index
        objects.append(dict(id=f'S{index}',name=row['name'],description=row['name'],is_receiving_structure=True,box=dict(top=float(yy.min()*1000/h),left=float(xx.min()*1000/w),bottom=float((yy.max()+1)*1000/h),right=float((xx.max()+1)*1000/w))))
        index+=1
    return result,objects
