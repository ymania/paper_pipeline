from pathlib import Path
import argparse
import pycolmap as pc
from PIL import Image,ImageOps

def reconstruct(source, work):
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    for photo in sorted(source.iterdir()):
        if photo.suffix.lower() in (".jpg",".jpeg"):
            with Image.open(photo) as original:
                img = ImageOps.exif_transpose(original).convert("RGB")
                img.thumbnail((1200,1200))
                img.save(images/photo.name,quality=92)
    db = work / "features.db"
    extract = pc.FeatureExtractionOptions()
    extract.num_threads = 4
    extract.sift.max_num_features = 3000
    pc.extract_features(db,images,extraction_options=extract,device=pc.Device.cpu)
    match = pc.FeatureMatchingOptions()
    match.num_threads = 4
    pc.match_exhaustive(db,matching_options=match,device=pc.Device.cpu)
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    models=pc.incremental_mapping(db,images,sparse)
    best=max(models.values(),key=lambda m:m.num_reg_images())
    (work/"best").mkdir(exist_ok=True)
    best.write(work/"best")
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("work",type=Path)
    a=parser.parse_args()
    reconstruct(a.source,a.work)
