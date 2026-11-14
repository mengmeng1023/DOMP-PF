"""Frozen S11 loss and offline diagnostics; labels never enter candidate extraction."""
import math
import numpy as np
from scipy.ndimage import label, maximum_filter, uniform_filter
import torch
from torch.nn import functional as F


def dense_loss(logits, mask):
    """Equal region BCE plus per-image SoftIoU; explicit absent-region rules."""
    if logits.ndim!=4 or logits.shape[1]!=1 or logits.shape!=mask.shape or not logits.numel():
        raise ValueError('Expected matching nonempty B1HW tensors')
    if not torch.isfinite(logits).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Expected finite logits and binary mask')
    m=mask.to(logits.dtype); axes=(1,2,3); nf=m.sum(axes); nb=(1-m).sum(axes)
    fg=(F.softplus(-logits)*m).sum(axes)/nf.clamp_min(1)
    bg=(F.softplus(logits)*(1-m)).sum(axes)/nb.clamp_min(1)
    bce=torch.where(nf==0,bg,torch.where(nb==0,fg,.5*(fg+bg)))
    p=logits.sigmoid(); inter=(p*m).sum(axes)
    iou=1-(inter+1e-6)/(p.sum(axes)+nf-inter+1e-6)
    return (bce+torch.where(nf>0,iou,torch.zeros_like(iou))).mean()


def mask_centers(mask):
    mask=np.asarray(mask)
    if mask.ndim!=2 or not mask.size or not np.isfinite(mask).all():raise ValueError('Invalid mask')
    cc,count=label(mask!=0,structure=np.ones((3,3),np.uint8));centers=[]
    for i in range(1,count+1):
        yy,xx=np.nonzero(cc==i);centers.append((float(xx.mean()),float(yy.mean())))
    centers.sort(key=lambda c:(c[1],c[0]))
    return np.asarray(centers,dtype=np.float64).reshape(-1,2)


def smooth_score(score):
    score=np.asarray(score,dtype=np.float64)
    if score.ndim!=2 or not score.size or not np.isfinite(score).all() or (score<0).any() or (score>1).any():
        raise ValueError('Expected finite unit HxW score')
    return np.clip(uniform_filter(score,size=3,mode='reflect'),0.,1.)


def extract_candidates(score, topk=20):
    """Pure score-only S6 algorithm, including flat/tied diagnostic candidates."""
    smoothed=smooth_score(score)
    if not isinstance(topk,(int,np.integer)) or topk<0:raise ValueError('Invalid topk')
    if not topk:return []
    maximum=maximum_filter(smoothed,size=7,mode='reflect');yy,xx=np.nonzero(smoothed==maximum)
    order=np.lexsort((xx,yy,-smoothed[yy,xx]));kept=[];result=[]
    for i in order:
        x,y=int(xx[i]),int(yy[i])
        if any(max(abs(x-kx),abs(y-ky))<=3 for kx,ky in kept):continue
        kept.append((x,y));result.append(dict(center_xy=[float(x),float(y)],grid_xy=[x,y],raw_score=float(smoothed[y,x]),valid=True,std_xy=None))
        if len(result)==topk:break
    return result


def is_degenerate(score):
    return bool(np.ptp(np.asarray(score,dtype=np.float64))<=1e-8 or np.ptp(smooth_score(score))<=1e-8)


def spatial_candidates(score):
    return [] if is_degenerate(score) else extract_candidates(score)


def match_candidates(candidates, centers, radius=5):
    centers=np.asarray(centers,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(centers).all() or not np.isfinite(radius) or radius<0:raise ValueError('Invalid centers/radius')
    assignments=np.full(len(candidates),-1,dtype=np.int64);target_order=np.lexsort((centers[:,0],centers[:,1]));used=set()
    order=sorted(range(len(candidates)),key=lambda i:(-candidates[i]['raw_score'],candidates[i].get('grid_xy',candidates[i]['center_xy'])[1],candidates[i].get('grid_xy',candidates[i]['center_xy'])[0],i))
    for ci in order:
        c=candidates[ci]
        if not c['valid']:continue
        center=np.asarray(c['center_xy'],dtype=np.float64)
        if center.shape!=(2,) or not np.isfinite(center).all():continue
        available=[i for i in target_order if i not in used]
        if not available:break
        squared=((centers[available]-center)**2).sum(axis=1);nearest=int(np.argmin(squared))
        if squared[nearest]<=radius**2:
            target=int(available[nearest]);assignments[ci]=target;used.add(target)
    return assignments


def top1_error(candidates, centers):
    valid=[c for c in candidates if c['valid']]
    if not valid or not len(centers):return None
    return float(np.sqrt(((np.asarray(centers)-valid[0]['center_xy'])**2).sum(axis=1)).min())


def summarize_rows(rows):
    denominator=sum(len(r['centers_xy']) for r in rows);matched={str(k):0 for k in (1,5,20)}
    strata={s:dict(targets=0,matched={str(k):0 for k in (1,5,20)}) for s in ('weak','strong')};errors=[];missing=0;empty=0
    for r in rows:
        centers=r['centers_xy'];cs=r['candidates'];error=top1_error(cs,centers)
        empty+=not bool(centers);missing+=bool(centers) and error is None
        if error is not None:errors.append(error)
        weak=r.get('weak',[False]*len(centers))
        for i in range(len(centers)):strata['weak' if weak[i] else 'strong']['targets']+=1
        for k in (1,5,20):
            assignments=match_candidates(cs[:k],centers);targets=set(int(v) for v in assignments if v>=0);matched[str(k)]+=len(targets)
            for i in targets:strata['weak' if weak[i] else 'strong']['matched'][str(k)]+=1
    for s in strata.values():s['recall']={k:v/s['targets'] if s['targets'] else None for k,v in s['matched'].items()}
    return dict(images=len(rows),targets=denominator,matched=matched,recall={k:v/denominator if denominator else None for k,v in matched.items()},top1_error_median_px=float(np.median(errors)) if errors else None,top1_error_p95_px=float(np.percentile(errors,95)) if errors else None,top1_error_count=len(errors),missing_top1=missing,empty_gt=empty,strata=strata)


def foreground90(foreground, background):
    fg=np.asarray(foreground).reshape(-1);bg=np.asarray(background).reshape(-1)
    if not fg.size or not bg.size or not np.isfinite(fg).all() or not np.isfinite(bg).all():raise ValueError('Need finite nonempty foreground/background')
    count=math.ceil(.9*len(fg));threshold=float(np.partition(fg,len(fg)-count)[len(fg)-count])
    fa=int((fg>=threshold).sum());ba=int((bg>=threshold).sum())
    return dict(threshold=threshold,requested_foreground_recall=.9,foreground_recall=fa/len(fg),background_activation=ba/len(bg),foreground_activated=fa,foreground_total=len(fg),background_activated=ba,background_total=len(bg),foreground_threshold_ties=int((fg==threshold).sum()),background_threshold_ties=int((bg==threshold).sum()),comparison='score >= threshold')


def selection_better(candidate, best):
    if best is None:return True
    def key(item):return (*[item['recall'][str(k)] for k in (20,5,1)],-item['step'])
    return key(candidate)>key(best)


def raw_contrast(gray, center):
    gray=np.asarray(gray,dtype=np.float64);yy,xx=np.indices(gray.shape);d2=(xx-center[0])**2+(yy-center[1])**2
    inner=gray[d2<=25];ring=gray[(d2>25)&(d2<=100)]
    if not len(inner) or not len(ring):return None
    return float((inner.mean()-ring.mean())/(ring.std()+1e-6))


def target_diagnostics(score, centers, candidates):
    score=np.asarray(score,dtype=np.float64);smooth=smooth_score(score);yy,xx=np.indices(score.shape);result=[]
    for center in centers:
        region=(xx-center[0])**2+(yy-center[1])**2<=25
        row={}
        for name,field in (('dense',score),('mean3',smooth)):
            maximum=float(field[region].max());positions=np.flatnonzero(region & (field==maximum));first=int(positions[0])
            row[name+'_max']=maximum;row[name+'_rank']=1+int((field>maximum).sum())+int((field.ravel()[:first]==maximum).sum())
        distance=[float(np.linalg.norm(np.asarray(c['center_xy'])-center)) for c in candidates]
        row['candidate_rank']=next((i+1 for i,v in enumerate(distance) if v<=5),None)
        row['nearest_candidate_rank']=int(np.argmin(distance))+1 if distance else None
        row['nearest_candidate_distance_px']=min(distance) if distance else None
        result.append(row)
    return result
