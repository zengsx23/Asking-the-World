"""Mask-guided original Cloth timelines for visual material comparisons."""
import cv2
import numpy as np

from atw.world_probing.engines.cloth_contact_time import contact_window
from atw.world_probing.evidence import write

NOTES = {
    'bendability': (
        'Compare resistance to bending and wrinkling through lifting, release and final draping. '
        'A cloth that retains broad smooth panels and resists wrapping is harder to bend; '
        'one that folds tightly and conforms around supports bends more easily. '
        'Use the complete sequence: early lifting can reverse the apparent ranking. '
        'Projected area changes also reflect visibility, rotation and folds, not just material response.'),
    'elasticity': (
        'Compare each cloth before interaction, after interaction and at the end. Greater elasticity means '
        'greater reversible stretch response, not greater stiffness. Follow changes in the same edges and '
        'visible surface features, allowing for folds, perspective and support motion. Wrinkles alone '
        'measure bending, not stretching. Compare the magnitude of the two cloths\' responses.')}
DISPLAY_NOTE = (
    'Original-video RGB panels with thin target-mask contours, one labeled identity per image. '
    'Both images use the same pixel scale and fixed crops across time. Labels give frame number and seconds; '
    'Retained 3D SWR cloth-to-rigid contact locates the observation window within the original clip; '
    'it is sampled fitted geometry, not a collision impulse log or proof of unloading. '
    'The last two panels retain the final state. Use the visible interaction, not fitted solver parameters.')


def sample_frames(boxes, contact, fps):
    gap=max(2,round(.2*fps));last=boxes[-1]['frame']
    if contact is not None:
        requested=[contact-2*gap,contact-gap,contact,contact+gap,contact+2*gap,last-gap,last]
    else:
        requested=np.linspace(boxes[0]['frame'],last,8)
    return sorted({min(boxes,key=lambda r:abs(r['frame']-f))['frame'] for f in requested})


def inspect(backend, object_ids, property_name, output):
    if any(backend.objects[name].get('physical_type')!='cloth' for name in object_ids):
        raise ValueError('Both material targets must be cloth identities')
    output.mkdir(parents=True,exist_ok=True)
    targets=[]
    for name in object_ids:
        boxes=[]
        for frame in range(backend.frame_count):
            try:mask=backend._mask(name,frame)
            except KeyError:continue
            y,x=np.where(mask)
            if not len(x):continue
            sx,sy=backend.width/mask.shape[1],backend.height/mask.shape[0]
            boxes.append(dict(frame=frame,area=len(x),box=[x.min()*sx,y.min()*sy,(x.max()+1)*sx,(y.max()+1)*sy]))
        if not boxes:raise ValueError(f'No observed cloth mask for {name}')
        contact=contact_window(backend.fit,name,backend.frame_count-1)
        selected=sample_frames(boxes,contact['first_contact_frame'],backend.fps)
        bounds=np.asarray([row['box'] for row in boxes if row['frame'] in selected])
        box=[bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max()]
        targets.append(dict(object_id=name,frames=selected,box=box,
            contact_timing=contact))
    size=int(np.ceil(max(max(t['box'][2]-t['box'][0],t['box'][3]-t['box'][1]) for t in targets)*1.15))
    capture=cv2.VideoCapture(str(backend.video));images=[]
    try:
        for index,target in enumerate(targets,1):
            name=target['object_id'];x0,y0,x1,y1=target['box']
            left=min(max(0,round((x0+x1-size)/2)),max(0,backend.width-size))
            top=min(max(0,round((y0+y1-size)/2)),max(0,backend.height-size))
            panels=[]
            for frame in target['frames']:
                capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
                if not ok:raise OSError(f'Cannot decode original frame {frame}')
                mask=cv2.resize(backend._mask(name,frame).astype(np.uint8),(backend.width,backend.height),interpolation=cv2.INTER_NEAREST)
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,contours,-1,(80,200,255),1)
                crop=image[top:top+size,left:left+size]
                crop=cv2.copyMakeBorder(crop,0,size-crop.shape[0],0,size-crop.shape[1],cv2.BORDER_CONSTANT,value=(25,25,25))
                panel=cv2.resize(crop,(360,360))
                panel=cv2.copyMakeBorder(panel,54,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
                for text,y in [(name,20),(f'f{frame}  {frame/backend.fps:.2f}s',44)]:
                    cv2.putText(panel,text,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                panels.append(panel)
            while len(panels)%4:panels.append(np.full_like(panels[0],25))
            path=output/f'target_{index:02d}.jpg'
            if not cv2.imwrite(str(path),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)])):
                raise OSError(f'Cannot write {path}')
            images.append(str(path))
    finally:capture.release()
    raw=dict(status='observed',property_name=property_name,object_ids=object_ids,targets=targets,
        crop_size_pixels=size,images=images,interval_seconds=[0,backend.end])
    write(output/'selection.json',raw)
    missing=[t['object_id'] for t in targets if t['contact_timing']['first_contact_frame'] is None]
    note=NOTES[property_name]+' '+DISPLAY_NOTE
    if missing:note+=' No fitted contact time was found for '+', '.join(missing)+'; those panels sample the visible clip instead.'
    return raw,note
