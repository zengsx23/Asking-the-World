"""Image-only local material segmentation with fixed instance seeds."""
import cv2
import numpy as np


def refine(raw, labels, rows, domain, wall):
    targets=[r for r in rows if r['is_receiving_structure']]
    protected=(labels!=0)&~np.isin(labels,[int(r['id'][1:]) for r in targets])
    lab=cv2.cvtColor(raw,cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[...,0]/=2.55
    result=labels.copy()
    result[~protected]=0
    best=np.full(labels.shape,np.inf)
    details=[]
    for row in targets:
        oid=int(row['id'][1:]);own=labels==oid
        yy,xx=np.where(own)
        pad=max(20,round(max(np.ptp(xx),np.ptp(yy))*.5))
        x,y,r,b=max(0,xx.min()-pad),max(0,yy.min()-pad),min(labels.shape[1],xx.max()+pad+1),min(labels.shape[0],yy.max()+pad+1)
        seed=own[y:b,x:r];image=raw[y:b,x:r];local=lab[y:b,x:r]
        allowed=domain[y:b,x:r]&~wall[y:b,x:r]&~protected[y:b,x:r]
        core=cv2.erode(seed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        values=local[core if core.any() else seed]
        cv2.setRNGSeed(41)
        _,groups,centers=cv2.kmeans(values,min(3,len(values)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.1),3,cv2.KMEANS_PP_CENTERS)
        center=centers[np.bincount(groups.ravel()).argmax()]
        distance=np.linalg.norm(local-center,axis=-1)
        outside=allowed&~cv2.dilate(seed.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
        bg_values=local[outside][::4]
        if len(bg_values)>3:
            _,bg_groups,bg_centers=cv2.kmeans(bg_values,3,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.1),3,cv2.KMEANS_PP_CENTERS)
            bg_center=bg_centers[np.bincount(bg_groups.ravel()).argmax()]
            bg_distance=np.linalg.norm(local-bg_center,axis=-1)
        else:
            bg_distance=np.full(seed.shape,100.)
        near=(distance<12)&allowed
        near_core=near&seed&(distance+3<bg_distance)
        grab=np.full(seed.shape,cv2.GC_PR_BGD,np.uint8)
        grab[distance<20]=cv2.GC_PR_FGD
        grab[bg_distance+3<distance]=cv2.GC_BGD
        grab[near_core]=cv2.GC_FGD
        grab[~allowed]=cv2.GC_BGD
        grab[[0,-1],:]=cv2.GC_BGD;grab[:,[0,-1]]=cv2.GC_BGD
        if np.any(grab==cv2.GC_FGD) and np.any(grab==cv2.GC_BGD):
            cv2.grabCut(image.copy(),grab,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
            mask=np.isin(grab,[cv2.GC_FGD,cv2.GC_PR_FGD])&allowed
            _,components=cv2.connectedComponents(mask.astype(np.uint8),8)
            keep=np.unique(components[near_core&mask]);keep=keep[keep!=0]
            mask &= np.isin(components,keep)
        else:
            mask=seed&allowed
        if np.linalg.norm(center[1:]-128)<12:
            extent=np.zeros_like(seed)
            sy,sx=np.where(seed)
            margin=max(2,round(labels.shape[1]/240))
            extent[max(0,sy.min()-margin):sy.max()+margin+1,max(0,sx.min()-margin):sx.max()+margin+1]=True
            mask=(seed | (mask&(distance<6)&extent))&allowed
            agreement=(seed&mask).sum()/max(1,(seed|mask).sum())
            if agreement>=.9:
                mask=seed&allowed
        seed_distance=cv2.distanceTransform((~seed).astype(np.uint8),cv2.DIST_L2,5)
        score=distance+.1*(960/labels.shape[1])*seed_distance
        take=mask&(score<best[y:b,x:r])
        result[y:b,x:r][take]=oid;best[y:b,x:r][take]=score[take]
        details.append(dict(id=row['id'],seed_area=int(seed.sum()),area=int(mask.sum()),lab_center=center.tolist()))
    return result,details
