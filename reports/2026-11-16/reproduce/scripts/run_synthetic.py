"""SYNTHETIC ONLY: controlled fusion stress test, no image inference or scene measurements."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation_gate import semantic_metrics
out=Path(__file__).parent;dst=out/'SYNTHETIC_ONLY';dst.mkdir(exist_ok=True)
x,y=np.meshgrid(np.linspace(0,12,60),np.linspace(0,8,40));points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
gt=np.zeros(len(points),int)
for cx in [2,6,10]:
 for cy in [2.4,5.7]:gt[(abs(points[:,0]-cx)<.7)&(abs(points[:,1]-cy)<1.)]=8
cameras=np.array([[x,4,-6.] for x in [-2,2,6,10,14]])
vectors=cameras[:,None,:]-points[None,:,:];weights=abs(vectors[:,:,2])/np.linalg.norm(vectors,axis=2)
# Ray intersection with two synthetic opaque panels at z=-1. Cameras are all at z=-6.
hits=cameras[:,None,:]+(5/6)*(points[None,:,:]-cameras[:,None,:])
occluded=((hits[:,:,0]>4.4)&(hits[:,:,0]<6.8)&(hits[:,:,1]<5.2))|((hits[:,:,0]>9.5)&(hits[:,:,0]<10)&(hits[:,:,1]>1))
faces=[]
for row in range(39):
 for col in range(59):
  a=row*60+col;faces += [(a,a+1,a+60),(a+1,a+61,a+60)]
with (dst/'synthetic_facade_surface.ply').open('w') as f:
 f.write(f'ply\nformat ascii 1.0\ncomment SYNTHETIC NOT PERSON HALL; arbitrary hypothetical units; plane surface not window holes\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
 for p in points:f.write(' '.join(map(str,p))+'\n')
 for t in faces:f.write('3 '+' '.join(map(str,t))+'\n')

def fuse(labels,w,mask=None,threshold=0):
 if mask is None:mask=np.ones_like(labels,bool)
 totals=np.stack([np.sum(w*mask*(labels==c),axis=0) for c in [0,8,150]],axis=1)
 pred=np.array([0,8,150])[totals.argmax(1)];mass=totals.sum(1)
 consensus=np.divide(totals.max(1),mass,out=np.zeros(len(gt)),where=mass>0)
 pred[(mass==0)|((mask.sum(0)<2)&(threshold>0))|(consensus<threshold)]=-1
 return pred

def score(pred):
 m=semantic_metrics(gt,pred);ious={str(c):m['per_class'].get(str(c),{}).get('iou') for c in [0,8,150]}
 return dict(coverage=float(np.mean(pred>=0)),window_iou=ious['8'],wall_iou=ious['0'],macro_iou=(ious['0']+ious['8'])/2,
  retained_accuracy=float(np.mean(gt[pred>=0]==pred[pred>=0])) if np.any(pred>=0) else None,
  confusion=[[int(np.sum((gt==c)&(pred==d))) for d in [0,8,150,-1]] for c in [0,8]])

settings=[('independent_errors',0.,0.05),('shared_window_error',1.,0.05),('unreliable_visibility',0.,0.35)]
results=[];detailed=[]
for regime,shared,viserr in settings:
 for seed in range(20):
  rng=np.random.default_rng(20261111+seed)
  labels=np.broadcast_to(gt,(5,len(gt))).copy()
  flip=rng.random(labels.shape)<np.where(gt==8,.25,.10)[None,:]
  labels[flip]=np.where(labels[flip]==8,0,8)
  labels[occluded]=150
  if shared:labels[:,gt==8]=0
  estimated_visible=(~occluded)^(rng.random(labels.shape)<viserr)
  methods={'single':labels[2], 'majority':fuse(labels,np.ones_like(weights)), 'geometry':fuse(labels,weights),
   'estimated_visibility':fuse(labels,weights,estimated_visible),
   'visibility_and_reject':fuse(labels,weights,estimated_visible,.75),
   'oracle_visibility_diagnostic':fuse(labels,weights,~occluded)}
  for method,pred in methods.items():
   s=score(pred);record={'regime':regime,'seed':seed,'method':method,**s};detailed.append(record)
   results.append({k:v for k,v in record.items() if k!='confusion'})
  np.savez_compressed(dst/f'{regime}_{seed:02d}.npz',points=points,truth=gt,cameras=cameras,true_occlusion=occluded,estimated_visible=estimated_visible,input_labels=labels,angle_weights=weights,**{f'pred_{k}':v for k,v in methods.items()})
with (dst/'runs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
summary=[]
for regime,_,_ in settings:
 for method in methods:
  rs=[r for r in results if r['regime']==regime and r['method']==method]
  row={'regime':regime,'method':method}
  for metric in ['window_iou','macro_iou','coverage','retained_accuracy']:
   vs=[r[metric] for r in rs if r[metric] is not None];row[metric+'_mean']=float(np.mean(vs));row[metric+'_sd']=float(np.std(vs,ddof=1))
  summary.append(row)
(dst/'summary.json').write_text(json.dumps(summary,indent=2));(dst/'confusion_matrices.json').write_text(json.dumps(detailed,indent=2))
config={'purpose':'SYNTHETIC_ONLY_MECHANISM_TEST_NOT_REAL_PERFORMANCE','unit':'hypothetical coordinate unit, not measured meters','geometry':{'plane_size':[12,8],'grid':[60,40],'windows':6,'window_size':[1.4,2.0],'cameras':5,'opaque_panels':2},'seeds':list(range(20261111,20261131)), 'regimes':settings,'visible_window_flip':.25,'visible_wall_flip':.10,'reject_consensus':.75,'minimum_accepted_views':2,'class_ids':{'wall':0,'window':8,'occluder':150,'abstain':-1},'metrics':'macro IoU fixed GT classes wall/window; abstain counts FN; coverage all points; retained accuracy only accepted points; SD across noise seeds is not uncertainty over buildings','limits':['No image formation or learned model inference','Analytic visibility is simulated; noisy visibility is artificial, not an implemented depth estimator','Normals and correspondences exact; no camera noise','Only one synthetic geometry, 20 seeds are NOT 20 buildings','Oracle uses synthetic hidden truth and cannot be deployed','Parameters are illustrative assumptions, not estimates from real data','Threshold fixed before runs, no tuning or held-out generalization claim','Plane grid is a triangle surface, not complete building or wall with physical window openings']}
(dst/'assumptions.json').write_text(json.dumps(config,indent=2));(dst/'checks.json').write_text(json.dumps({'finite_points':bool(np.isfinite(points).all()),'triangle_indices_valid':bool(np.max(faces)<len(points)),'gt_window_points':int(np.sum(gt==8)),'vertices':len(points),'triangles':len(faces),'records':len(results),'shared_failure_all_window_iou_zero':all(r['window_iou']==0 for r in results if r['regime']=='shared_window_error')},indent=2))
(dst/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir() if p.name!='manifest.json'},indent=2))
print(json.dumps(summary,indent=2))
