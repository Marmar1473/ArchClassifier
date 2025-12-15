import os
import cv2
from torch.utils.data import Dataset
from .config import CLASS_NAMES

class ArchitectureDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        split: 'train', 'val' или 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_map = {name: i for i, name in enumerate(CLASS_NAMES)}

        for class_name in CLASS_NAMES:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(img_path)
                    self.labels.append(self.class_map[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label