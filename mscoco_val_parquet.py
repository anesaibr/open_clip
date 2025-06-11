# make_mscoco_val_parquet.py
import json
import pandas as pd

# 1) Load the COCO val2017 captions JSON
ann_path = "annotations/captions_val2017.json"
with open(ann_path, "r") as f:
    data = json.load(f)

# 2) Each entry in data["annotations"] has fields:
#      - "image_id"  (int, e.g. 391895)
#      - "caption"   (string)
#
#    We'll build one row per annotation, with the URL and the caption text.
rows = []
for ann in data["annotations"]:
    image_id = ann["image_id"]
    caption  = ann["caption"]
    # Build the URL for that image_id in val2017. COCO uses 12-digit zero-padding:
    img_filename = f"{image_id:012d}.jpg"
    url = f"http://images.cocodataset.org/val2017/{img_filename}"
    rows.append({"URL": url, "TEXT": caption})

# 3) Make a DataFrame and save to Parquet
df = pd.DataFrame(rows)
print(f"Total rows (i.e. image-caption pairs): {len(df)}")  # should print 25000

# 4) Write to a Parquet file (no index column)
out_parquet = "mscoco_val.parquet"
df.to_parquet(out_parquet, index=False)
print(f"Wrote {out_parquet} ({df.shape[0]}×{df.shape[1]})")
