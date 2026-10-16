"""Numerical semantic readouts with explicit unavailable measurements."""
import cv2
import numpy as np


def longest_run(values):
    edges=np.diff(np.r_[False,values,False].astype(np.int8))
    return int(max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1),default=0))


def fluid_static(xy,births,mask,first,last,radius):
    height,width=mask.shape
    xy=np.asarray(xy)[first:last+1]
    frames=np.arange(first,last+1)
    active=frames[:,None]>=births[None,:]
    finite=np.isfinite(xy).all(-1)
    x=np.rint(np.nan_to_num(xy[...,0])).astype(int)
    y=np.rint(np.nan_to_num(xy[...,1])).astype(int)
    valid=active&finite&(x>=0)&(x<width)&(y>=0)&(y<height)
    xclip,yclip=x.clip(0,width-1),y.clip(0,height-1)
    denominator=int((births<=last).sum())
    if not denominator or not mask.any():
        raise ValueError('No born target particles or usable static mask')
    ys,xs=np.where(mask)
    interior=np.zeros_like(mask)
    for row in range(int(ys.min()),int(ys.max())+1):
        occupied=np.flatnonzero(mask[row])
        if len(occupied)>=2:
            interior[row,occupied[0]:occupied[-1]+1]=~mask[row,occupied[0]:occupied[-1]+1]
    inside=valid&interior[yclip,xclip]
    top=np.full(width,np.nan);top[xs]=height
    np.minimum.at(top,xs,ys)
    top_mask=np.zeros_like(mask,dtype=np.uint8)
    columns=np.flatnonzero(np.isfinite(top))
    top_mask[top[columns].astype(int),columns]=1
    distance=cv2.distanceTransform(1-top_mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    above=xy[...,1]<=top[xclip]+.5
    away=(xy[...,0]>=xs.min()+radius)&(xy[...,0]<=xs.max()-radius)
    contacts=[]
    for factor in (1,2,3):
        hit=valid&above&away&(distance[yclip,xclip]<=factor*radius+.5)
        contacts.append(dict(tolerance_pixels=float(factor*radius+.5),
            unique_particles=int(hit.any(0).sum()),fraction=float(hit.any(0).sum()/denominator),
            peak_particles=int(hit.sum(1).max(initial=0)),sustained_frames=longest_run(hit.sum(1)>=2)))
    return dict(born_particles=denominator,interior_pixels=int(interior.sum()),
        container_status='measured' if interior.any() else 'insufficient_geometry',
        entered_fraction=float(inside.any(0).sum()/denominator) if interior.any() else None,
        final_fraction=float(inside[-1].sum()/denominator) if interior.any() else None,
        upper_surface_contact=contacts)


def pit_crossings(points,frames,gap,height,fps):
    """Detect direct entry or whole-body descent after rolling into the mouth."""
    points=np.asarray(points);frames=np.asarray(frames)
    finite=np.isfinite(points).all((-1,-2))
    if not len(points) or not finite.all():
        return dict(downward_entry_times_seconds=[],fully_below_image=None,coverage_complete=False)
    center=points.mean(1);top=points[...,1].min(1)
    x0,y0,x1,_=gap
    entries=[]
    for i in range(1,len(points)):
        if frames[i]!=frames[i-1]+1 or not center[i-1,1]<=y0<center[i,1]:
            continue
        fraction=(y0-center[i-1,1])/(center[i,1]-center[i-1,1])
        x=center[i-1,0]+fraction*(center[i,0]-center[i-1,0])
        if x0<x<x1 and np.all(center[i:,1]>=y0):
            entries.append(float((frames[i-1]+fraction)/fps))
    if not entries:
        # A rolling body's center can already be below the projected mouth line.
        # Its top crossing still establishes downward passage through that mouth;
        # a body arriving wholly below it cannot produce this crossing.
        for i in range(1,len(points)):
            if frames[i]!=frames[i-1]+1 or not top[i-1]<=y0<top[i]:
                continue
            fraction=(y0-top[i-1])/(top[i]-top[i-1])
            x=center[i-1,0]+fraction*(center[i,0]-center[i-1,0])
            if x0<x<x1 and np.all(center[i:,1]>=y0):
                entries.append(float((frames[i-1]+fraction)/fps))
    return dict(downward_entry_times_seconds=entries,fully_below_image=bool(top[-1]>=height),
                coverage_complete=bool(np.all(np.diff(frames)==1)))
