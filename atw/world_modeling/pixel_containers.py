"""Refine observed receiving structures with local material and conditional identity review."""
import numpy as np
from atw.world_modeling.scene_assets import write
from atw.world_modeling.pixel_container_material import refine as refine_material
from atw.world_modeling.pixel_container_identity import ambiguous, review

def refine(raw,labels,rows,proposals,raw_candidates,domain,requests,output,first_frame):
    output.mkdir();np.savez_compressed(output/'before.npz',labels=labels)
    write(output/'before_inventory.json',dict(objects=rows))
    result,details=refine_material(raw,labels,rows,domain,np.zeros_like(domain))
    reviewed=ambiguous(labels,rows,details)
    if reviewed:
        seeds,rows=review(raw,labels,rows,domain,requests,output,first_frame)
        np.savez_compressed(output/'reviewed_seeds.npz',labels=seeds)
        result,details=refine_material(raw,seeds,rows,domain,np.zeros_like(domain))
    final=np.zeros_like(labels);objects=[];mapping=[];h,w=labels.shape
    for row in rows:
        mask=result==int(row['id'][1:]);yy,xx=np.where(mask)
        if not len(xx):raise ValueError('Refinement produced empty instance '+row['id'])
        oid=f'S{len(objects)+1}';final[mask]=len(objects)+1
        obj={k:v for k,v in row.items() if k!='reviewed_box'}
        objects.append(dict(obj,id=oid,box=dict(top=float(yy.min()*1000/h),left=float(xx.min()*1000/w),bottom=float((yy.max()+1)*1000/h),right=float((xx.max()+1)*1000/w))))
        mapping.append(dict(previous_id=row['id'],final_id=oid))
    write(output/'summary.json',dict(status='refined',method='local_material',identity_review=reviewed,details=details,final_id_mapping=mapping))
    return final,objects
