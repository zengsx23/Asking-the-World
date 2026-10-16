"""Original-video material timelines selected from observed masks only."""
import cv2
import numpy as np

from atw.world_probing.evidence import write

MATERIAL_NOTE = 'Compare each target with its own earliest clear shape, then inspect deformation and later recovery across the clip. Elasticity here requires visible deformation and substantial recovery; plasticity requires substantial residual deformation after unloading. Little deformation alone supports neither strong elasticity nor strong plasticity. Compare the degree of response, not just whether both bodies deform.\nA round silhouette does not by itself prove full recovery: compare shape and apparent extent with the initial reference, allowing for perspective, rotation and occlusion. Do not infer volume loss from pixel area alone. A later frame may contain another impact or sustained pressure; check nearby obstacles and other bodies before calling a deformation permanent. Use multiple clear views when available. Rebound height or travel distance alone does not rank these material properties.\nThe panels are original-video observations selected across time; selection labels are not material classifications or proof of unloading. Judge the literal comparison from the visible response, not fitted stiffness or simulated motion. If a necessary response cannot be observed, identify the missing evidence. Each image follows one labeled target through the clip. All panels within an image use the same pixel scale; the two target images may use different scales. Thin contours mark the observed mask. Frames are chronological, with original frame numbers and seconds.'

def target_samples(backend,oid):
    measurements=[]
    for f in range(backend.frame_count):
        try:mask=backend._mask(oid,f)
        except KeyError:continue
        yy,xx=np.where(mask)
        if len(xx)<2:continue
        sx,sy=backend.width/mask.shape[1],backend.height/mask.shape[0]
        points=np.column_stack([xx*sx,yy*sy])
        eig=np.linalg.eigvalsh(np.cov(points.T))
        aspect=float(np.sqrt(max(eig[-1],1)/max(eig[0],1)))
        box=[float(xx.min()*sx),float(yy.min()*sy),float((xx.max()+1)*sx),float((yy.max()+1)*sy)]
        measurements.append(dict(frame=f,area=float(len(xx)*sx*sy),aspect=aspect,box=box,
            clipped=bool(xx.min()==0 or yy.min()==0 or xx.max()==mask.shape[1]-1 or yy.max()==mask.shape[0]-1)))
    visible=[r for r in measurements if not r['clipped']]
    if not visible:raise ValueError(f'No complete frame for {oid}')
    chosen={}
    def add(row,label):chosen.setdefault(row['frame'],[]).append(label)
    # Eight evenly spaced visible observations retain initial and late states.
    for t in np.linspace(visible[0]['frame'],visible[-1]['frame'],8):
        add(min(visible,key=lambda r:abs(r['frame']-t)),'timeline')
    # Two temporally separated elongated silhouettes, with subsequent observations.
    # Elongation is only a frame-selection cue; it can also arise from occlusion.
    peaks=[]
    for row in sorted(visible,key=lambda r:r['aspect'],reverse=True):
        if all(abs(row['frame']-p['frame'])>=round(.4*backend.fps) for p in peaks):
            peaks.append(row)
        if len(peaks)==2:break
    for row in peaks:
        add(row,'shape_sample')
        later=[r for r in visible if r['frame']>=row['frame']+round(.3*backend.fps)]
        if later:add(later[0],'later_sample')
    frames=sorted(chosen)
    lookup={r['frame']:r for r in measurements}
    size=int(np.ceil(max(max(r['box'][2]-r['box'][0],r['box'][3]-r['box'][1]) for r in visible)*2.8))
    return measurements,[(lookup[f],chosen[f]) for f in frames],size

def sheet(backend,oid,selected,size,path):
    capture=cv2.VideoCapture(str(backend.video))
    panels=[]
    try:
        for row,_ in selected:
            f=row['frame'];capture.set(cv2.CAP_PROP_POS_FRAMES,f)
            ok,im=capture.read()
            if not ok:raise RuntimeError(f'Cannot decode frame {f}')
            mask=cv2.resize(backend._mask(oid,f).astype(np.uint8),(backend.width,backend.height),interpolation=cv2.INTER_NEAREST)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im,contours,-1,(80,200,255),1)
            x0,y0,x1,y1=row['box'];cx,cy=(x0+x1)/2,(y0+y1)/2
            left=min(max(0,round(cx-size/2)),max(0,backend.width-size))
            top=min(max(0,round(cy-size/2)),max(0,backend.height-size))
            crop=im[top:top+size,left:left+size]
            # Pad large crops rather than stretching a rectangular image.
            crop=cv2.copyMakeBorder(crop,0,size-crop.shape[0],0,size-crop.shape[1],
                cv2.BORDER_CONSTANT,value=(25,25,25))
            canvas=cv2.resize(crop,(300,300))
            canvas=cv2.copyMakeBorder(canvas,56,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            cv2.putText(canvas,oid,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1)
            cv2.putText(canvas,f'f{f}  {f/backend.fps:.2f}s',(8,44),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1)
            panels.append(canvas)
    finally:
        capture.release()
    while len(panels)%4:panels.append(np.full_like(panels[0],25))
    grid=np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)])
    if not cv2.imwrite(str(path),grid,[cv2.IMWRITE_JPEG_QUALITY,95]):
        raise OSError(f'Cannot save observation image: {path}')

def inspect(backend, object_ids, property_name, output):
    output.mkdir(parents=True, exist_ok=True)
    images, objects = [], []
    for index, oid in enumerate(object_ids, 1):
        measurements, selected, size = target_samples(backend, oid)
        path = output / f'target_{index:02d}.jpg'
        sheet(backend, oid, selected, size, path)
        write(output / f'target_{index:02d}_selection.json', dict(
            object_id=oid, measurements=measurements, selected=selected, crop_size_pixels=size))
        images.append(str(path))
        objects.append(dict(object_id=oid, frames=[row['frame'] for row, _ in selected]))
    return dict(status='observed', property_name=property_name, objects=objects,
                interval_seconds=[0, backend.end], images=images)
