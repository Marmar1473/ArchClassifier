import os
import cv2

BASE_DIR = "dataset"

bad_files = []

for split in ["train", "val", "test"]:
    split_path = os.path.join(BASE_DIR, split)

    if not os.path.isdir(split_path):
        continue

    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)

            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            if img is None:
                bad_files.append({
                    "split": split,
                    "class": class_name,
                    "filename": filename,
                    "path": file_path
                })

print(f"Найдено нечитаемых файлов: {len(bad_files)}\n")

for item in bad_files:
    print(
        f"[{item['split']} / {item['class']}]  "
        f"{item['filename']}\n"
        f" → {item['path']}\n"
    )