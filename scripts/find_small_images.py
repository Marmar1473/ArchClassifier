from PIL import Image
import os

min_size = 128
cls_path = os.path.join("dataset", "train", "Minimalism")

for f in os.listdir(cls_path):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(cls_path, f)
        img = Image.open(path)
        if img.width < min_size or img.height < min_size:
            print("TO REMOVE:", path)