"""Associate predicted image boxes using existing SfM tracks, without truth labels."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.colmap_io import read_images_txt,read_points3D_txt
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
if (out/'associations.json').exists():raise ValueError('Refuse overwrite')
model=a.model
images=read_images_txt(str(model/'images.txt'));byname={v['name']:v for v in images.values()}
xyz,_,tracks=read_points3D_txt(str(model/'points3D.txt'));coords={pid:xyz[j] for j,pid in enumerate(tracks)}
data=json.loads((out/'detections.json').read_text());nodes=[]
for row in data['images']:
 for c in row['candidates']:
  x0,y0,x1,y1=c['box_xyxy'];ids={pid for x,y,pid in byname[row['image']]['points2d'] if x0<=x<=x1 and y0<=y<=y1}
  nodes.append({**c,'image':row['image'],'track_ids':sorted(ids)})
edges=[]
for i,n in enumerate(nodes):
 for j in range(i):
  m=nodes[j]
  if n['image']==m['image']:continue
  s=set(n['track_ids']);t=set(m['track_ids']);common=len(s&t);ratio=common/min(len(s),len(t)) if s and t else 0
  if common>=3 and ratio>=.20:edges.append((ratio,common,j,i))
parent=list(range(len(nodes)));members={i:{i} for i in parent};accepted=[];rejected=[]
def find(i):
 while parent[i]!=i:i=parent[i]
 return i
for ratio,common,i,j in sorted(edges,key=lambda x:(-x[0],-x[1],x[2],x[3])):
 x,y=find(i),find(j)
 if x==y:continue
 if {nodes[k]['image'] for k in members[x]} & {nodes[k]['image'] for k in members[y]}:
  rejected.append([i,j,'same-view conflict']);continue
 parent[y]=x;members[x]|=members.pop(y);accepted.append({'node_a':i,'node_b':j,'shared_tracks':common,'overlap':ratio})
groups=[]
for k,indices in enumerate(members.values()):
 ns=[nodes[i] for i in sorted(indices)];pids=sorted(set().union(*(set(n['track_ids']) for n in ns)));q=np.array([coords[pid] for pid in pids])
 groups.append({'group_id':k,'status':'multi_view_candidate' if len(ns)>=3 else 'insufficient_views','views':len(ns),'node_indices':sorted(indices),'candidate_ids':[n['id'] for n in ns],'track_ids':pids,'support_centroid':q.mean(0).tolist() if len(q) else None,'support_bounds':[q.min(0).tolist(),q.max(0).tolist()] if len(q) else None})
assert all(len(g['node_indices'])==len({nodes[i]['image'] for i in g['node_indices']}) for g in groups)
report={'source':'model_prediction','parameters':{'min_shared_tracks':3,'min_overlap_smaller_set':.20,'min_support_views':3},'nodes':nodes,'accepted_edges':accepted,'conflict_edges':rejected,'groups':groups,'evidence_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'detections.json',model/'images.txt',model/'points3D.txt']},'limitations':['track overlap associates BOXES, not verified window instances','box tracks may belong to frame, wall or occluder','centroid/bounds describe track support only, NOT measured window geometry','no depth-based occlusion rejection; track visibility does not certify every box pixel','same-scene development experiment; no instance accuracy claimed','no manual annotation consumed; grouping thresholds illustrative, not tuned on truth']}
(out/'associations.json').write_text(json.dumps(report,indent=2));print(json.dumps({'detections':len(nodes),'groups':len(groups),'supported':sum(g['status']=='multi_view_candidate' for g in groups),'views':[g['views'] for g in groups]},indent=2))
