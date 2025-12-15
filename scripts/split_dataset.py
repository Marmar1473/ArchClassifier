import os
import random
import shutil

random.seed(42)

root = "dataset"

class_name = "Minimalism"

cls_path = os.path.join(root, class_name)

splits = {"train": 0.7, "val": 0.15, "test": 0.15}

files = [f for f in os.listdir(cls_path)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))
         and os.path.isfile(os.path.join(cls_path, f))]

random.shuffle(files)

n = len(files)
i = 0

for split, frac in splits.items():
    split_path = os.path.join(root, split, class_name)
    os.makedirs(split_path, exist_ok=True)

    k = int(frac * n)
    for f in files[i:i+k]:
        shutil.move(os.path.join(cls_path, f), split_path)

    i += k

for f in files[i:]:
    shutil.move(os.path.join(cls_path, f), os.path.join(root, "train", class_name))