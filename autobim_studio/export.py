import argparse, ast, json, sys
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import DBSCAN
import ifcopenshell, ifcopenshell.api as api, ifcopenshell.geom, ifcopenshell.validate
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.weighted_voting import weighted_vote

def fuse(work):
    rec=pycolmap.Reconstruction(work/"best")
    points=list(rec.points3D.values())
    xyz=np.array([p.xyz for p in points])
    neighbors=cKDTree(xyz).query(xyz,k=min(10,len(xyz)))[1]
    local=xyz[neighbors]
    local-=local.mean(axis=1,keepdims=True)
    normals=np.linalg.eigh(local.transpose(0,2,1)@local)[1][:,:,0]
    maps, indices, cameras={}, {}, {}
    for iid,im in rec.images.items():
        cam=rec.camera(im.camera_id)
        mask=Image.open(work/"labels"/(Path(im.name).stem+".png"))
        maps[iid]=np.array(mask.resize((cam.width,cam.height),Image.Resampling.NEAREST))
        indices[iid]=len(indices)
        cameras[iid]={"C":im.projection_center()}
    vis={}
    for i,pt in enumerate(points):
        xy={e.image_id:rec.image(e.image_id).points2D[e.point2D_idx].xy for e in pt.track.elements}
        vis[i]={"cam_indices":list(xy),"px_coords":list(xy.values()),"depths":[]}
    labels,conf=weighted_vote(xyz,normals,vis,{},cameras,list(maps.values()),indices)
    return rec,xyz,labels,conf

def export(work,out):
    rec,xyz,labels,confidence=fuse(work)
    mapping=json.loads((ROOT/"config/label_to_ifc.json").read_text())
    tree=ast.parse((ROOT/"scripts/2_segformer_infer.py").read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")=="ADE20K_CLASSES")
    model=api.run("project.create_file",version="IFC4")
    def entity(cls,name):
        return api.run("root.create_entity",model,ifc_class=cls,name=name)
    project=entity("IfcProject","AutoBIM - uncalibrated reconstruction")
    unit=api.run("unit.add_si_unit",model,unit_type="LENGTHUNIT")
    api.run("unit.assign_unit",model,units=[unit])
    site=entity("IfcSite","Photo survey")
    api.run("aggregate.assign_object",model,products=[site],relating_object=project)
    ctx=api.run("context.add_context",model,context_type="Model")
    body=api.run("context.add_context",model,context_type="Model",context_identifier="Body",target_view="MODEL_VIEW",parent=ctx)
    spacing=float(np.median(cKDTree(xyz).query(xyz,k=min(5,len(xyz)))[0][:,-1]))
    counts={}
    represented=0
    for label in sorted(set(labels)-{-1}):
        idx=np.flatnonzero(labels==label)
        clusters=DBSCAN(eps=spacing*4,min_samples=6).fit_predict(xyz[idx])
        name=names[int(label)]
        for ci in sorted(set(clusters)-{-1}):
            ids=idx[clusters==ci]
            pts=xyz[ids]
            if np.linalg.matrix_rank(pts-pts.mean(axis=0))<3:
                continue
            hull=ConvexHull(pts)
            faces=hull.simplices.copy()
            for j,(a,b,c) in enumerate(faces):
                if np.dot(np.cross(pts[b]-pts[a],pts[c]-pts[a]),hull.equations[j,:3])<0:
                    faces[j]=[a,c,b]
            used,remap=np.unique(faces,return_inverse=True)
            obj=entity(mapping.get("window" if name=="windowpane" else name,"IfcBuildingElementProxy"),f"{name} {ci+1}")
            api.run("spatial.assign_container",model,products=[obj],relating_structure=site)
            api.run("geometry.edit_object_placement",model,product=obj)
            rep=api.run("geometry.add_mesh_representation",model,context=body,vertices=[pts[used].tolist()],faces=[remap.reshape(-1,3).tolist()])
            api.run("geometry.assign_representation",model,product=obj,representation=rep)
            props=api.run("pset.add_pset",model,product=obj,name="AutoBIM_Semantics")
            api.run("pset.edit_pset",model,pset=props,properties={"ADE20KClass":name,"SourcePoints":int(len(ids)),"MeanVoteShare":float(confidence[ids].mean()),"Geometry":"Convex hull of segmented sparse points","Scale":"Local SfM units; metric scale unverified"})
            counts[name]=counts.get(name,0)+1
            represented+=len(ids)
    assert counts
    model.write(str(out))
    check=ifcopenshell.open(str(out))
    logger=ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(check,logger,express_rules=True)
    errors=[e["message"] for e in logger.statements if e["level"]=="error"]
    shapes=[ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(),e) for e in check.by_type("IfcElement")]
    assert all(len(s.geometry.faces)>0 for s in shapes)
    report=dict(schema=check.schema,registered_images=rec.num_reg_images(),points=len(xyz),clustered_points=represented,classes=counts,geometry_checked=len(shapes),validation_errors=errors)
    out.with_suffix(".json").write_text(json.dumps(report,indent=2))
    print(report)
    assert not errors
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("work",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    export(a.work,a.out)

