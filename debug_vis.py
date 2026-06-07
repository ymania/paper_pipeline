"""Debug: 检查投影可见性"""
import pickle
import numpy as np

tracks = pickle.load(open('output/intermediate/tracks.pkl', 'rb'))
images = pickle.load(open('output/intermediate/images.pkl', 'rb'))
visibility = pickle.load(open('output/intermediate/visibility.pkl', 'rb'))

# Track stats
track_lengths = [len(v) for v in tracks.values()]
print(f"Tracks: {len(tracks)} points, avg track len: {np.mean(track_lengths):.2f}, max: {max(track_lengths)}")

# Show first 3 points
sample_keys = list(tracks.keys())[:3]
for k in sample_keys:
    img_ids = [t[0] for t in tracks[k]]
    print(f"  COLMAP point {k}: visible in images {img_ids}")

# All unique image IDs in tracks
all_track_imgs = set()
for v in tracks.values():
    for t in v:
        all_track_imgs.add(t[0])
print(f"Image IDs in tracks: {sorted(all_track_imgs)}")
print(f"Image IDs in images dict: {sorted(images.keys())}")

# Visibility stats
vis_counts = [len(v['cam_indices']) for v in visibility.values()]
print(f"\nVisibility: {len(visibility)} points")
print(f"  Points with >0 views: {sum(1 for c in vis_counts if c > 0)}")
print(f"  Points with 0 views: {sum(1 for c in vis_counts if c == 0)}")

# Check random point with >0 views
for idx, vis in visibility.items():
    if len(vis['cam_indices']) > 0:
        print(f"\n  Point {idx}: {len(vis['cam_indices'])} views in images {vis['cam_indices']}")
        print(f"  Track for this point: {tracks.get(idx, 'NOT IN TRACKS')}")
        break

# Check: do visibility keys match track keys?
vis_keys = set(visibility.keys())
track_keys = set(tracks.keys())
print(f"\nVisibility keys: 0..{max(vis_keys)}, count={len(vis_keys)}")
print(f"Track keys: min={min(track_keys)}, max={max(track_keys)}, count={len(track_keys)}")
print(f"Intersection: {len(vis_keys & track_keys)}")
