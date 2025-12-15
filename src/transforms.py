import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import IMAGE_SIZE

def get_transforms(stage="train"):
    if stage == "train":
        return A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
            A.GaussianBlur(blur_limit=(3, 7), p=0.05),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.1),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.2),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])