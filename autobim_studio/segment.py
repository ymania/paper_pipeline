from pathlib import Path
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

def segment(images,model):
    out=images.parent/"labels"
    out.mkdir(exist_ok=True)
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),options,providers=["CPUExecutionProvider"])
    mean=np.array([.485,.456,.406],dtype=np.float32)
    std=np.array([.229,.224,.225],dtype=np.float32)
    for path in sorted(images.iterdir()):
        with Image.open(path) as im:
            rgb=np.asarray(im.convert("RGB").resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255
        batch=((rgb-mean)/std).transpose(2,0,1)[None]
        logits=session.run(None,{session.get_inputs()[0].name:batch})[0][0]
        Image.fromarray(logits.argmax(axis=0).astype("uint8")).save(out/(path.stem+".png"))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("images",type=Path)
    p.add_argument("model",type=Path)
    a=p.parse_args()
    segment(a.images,a.model)
