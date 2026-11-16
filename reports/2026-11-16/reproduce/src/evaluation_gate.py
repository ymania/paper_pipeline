"""Evaluation contracts. Metadata is an attestation, not proof of label correctness."""
import numpy as np

def validate_annotations(data, dataset_id, point_ids):
    if data.get('dataset_id') != dataset_id:
        raise ValueError('Annotation dataset fingerprint mismatch')
    if data.get('source') not in ('human_blind_review','independent_public_ground_truth'):
        raise ValueError('Independent ground truth required; predictions/assistant drafts are not ground truth')
    if not data.get('reviewer') or data.get('independent_review_confirmed') is not True:
        raise ValueError('Independent reviewer attestation required')
    allowed=set(map(int,point_ids));seen=set();selected=[]
    for s in data.get('samples',[]):
        pid=s['point3D_id']
        if type(pid) is not int or pid not in allowed or pid in seen:
            raise ValueError('Duplicate or unknown point ID')
        seen.add(pid);label=s.get('label')
        if label is None:continue
        if type(label) is not int or not -1<=label<=150:
            raise ValueError('Invalid label; unknown=-1, ADE20K=0..149, other=150')
        if label>=0:selected.append((pid,label))
    if not selected:raise ValueError('No independently annotated evaluable samples')
    return selected

def semantic_metrics(truth, prediction):
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    if truth.shape!=prediction.shape or truth.ndim!=1 or not len(truth):
        raise ValueError('Nonempty paired one-dimensional arrays required')
    if not np.issubdtype(truth.dtype,np.integer) or not np.issubdtype(prediction.dtype,np.integer):
        raise ValueError('Integer class IDs required')
    if np.any(truth<0) or np.any(truth>150) or np.any(prediction<-1) or np.any(prediction>150):
        raise ValueError('Invalid labels; unknown GT must be excluded explicitly before scoring')
    classes=sorted(set(truth.tolist())|set(prediction[prediction>=0].tolist()))
    per={}
    for c in classes:
        tp=int(np.sum((truth==c)&(prediction==c)));fp=int(np.sum((truth!=c)&(prediction==c)));fn=int(np.sum((truth==c)&(prediction!=c)))
        per[str(c)]={'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn),
                     'precision':tp/(tp+fp) if tp+fp else None,
                     'recall':tp/(tp+fn) if tp+fn else None}
    return {'n':len(truth),'accuracy':float(np.mean(truth==prediction)),
            'coverage':float(np.mean(prediction>=0)),
            'macro_iou':float(np.mean([v['iou'] for v in per.values()])),
            'per_class':per,'class_policy':'union of GT and non-abstaining predictions'}
