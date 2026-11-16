"""Frozen prompt window candidates; labels are predictions, never human truth."""
from pathlib import Path
import argparse,json,time,hashlib
import torch
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--revision',required=True);a=p.parse_args();out=a.output
if (out/'detections.json').exists():raise ValueError('Refuse overwrite completed run')
out.mkdir(parents=True,exist_ok=True);torch.manual_seed(20261111)
model_id='IDEA-Research/grounding-dino-tiny';start=time.perf_counter()
processor=AutoProcessor.from_pretrained(model_id,revision=a.revision)
model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id,revision=a.revision).to('cuda').eval()
print('Loaded model',flush=True);load_seconds=time.perf_counter()-start
DATA=a.images
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
results=[]
for j in range(1118,1129):
 name=f'IMG_{j}.JPG';path=DATA/name;im=Image.open(path).convert('RGB');start=time.perf_counter()
 inputs=processor(images=im,text='a window.',return_tensors='pt').to('cuda')
 with torch.inference_mode():outputs=model(**inputs)
 raw=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=.30,text_threshold=.25,target_sizes=[(im.height,im.width)])[0]
 # Save all thresholded outputs; deterministic class-agnostic NMS for duplicate boxes.
 boxes=raw['boxes'].cpu().tolist();scores=raw['scores'].cpu().tolist();texts=raw.get('text_labels',[])
 def iou(b,c):
  inter=max(0,min(b[2],c[2])-max(b[0],c[0]))*max(0,min(b[3],c[3])-max(b[1],c[1]));den=(b[2]-b[0])*(b[3]-b[1])+(c[2]-c[0])*(c[3]-c[1])-inter
  return inter/den if den>0 else 0
 kept=[]
 for k in sorted(range(len(boxes)),key=lambda k:(-scores[k],k)):
  if all(iou(boxes[k],boxes[t])<=.5 for t in kept):kept.append(k)
 candidates=[dict(id=f'{j}_{k}',box_xyxy=boxes[k],score=scores[k],phrase=texts[k] if len(texts)>k else 'window') for k in kept]
 row=dict(image=name,size=list(im.size),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_boxes=boxes,raw_scores=scores,candidates=candidates,seconds=time.perf_counter()-start)
 results.append(row)
 canvas=im.copy();canvas.thumbnail((1200,800));d=ImageDraw.Draw(canvas);scale=canvas.width/im.width
 for c in candidates:
  box=[v*scale for v in c['box_xyxy']];d.rectangle(box,outline='#00e4d5',width=3);d.text((box[0],max(0,box[1]-26)),f"{c['id']} {c['score']:.2f}",font=font,fill='white',stroke_width=2,stroke_fill='black')
 canvas.save(out/f'{j}_candidates.jpg',quality=92)
 print(name,len(candidates),'candidates',round(row['seconds'],2),'sec',flush=True)
 (out/'progress.json').write_text(json.dumps(results,indent=2))
config=dict(model=model_id,revision=a.revision,prompt='a window.',box_threshold=.30,text_threshold=.25,nms_iou=.5,torch=torch.__version__,device=torch.cuda.get_device_name(),load_seconds=load_seconds,source='model_prediction',limitations=['boxes are not masks or window outlines','scores are uncalibrated','no human ground truth used in detection','development images, not held-out evaluation','candidate boxes may enclose occluders or wall'])
(out/'detections.json').write_text(json.dumps({'config':config,'images':results},indent=2));(out/'processor_config.json').write_text(json.dumps(processor.image_processor.to_dict(),indent=2,default=str))
