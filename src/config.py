import os
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')

MODEL_NAME = "efficientnet_v2_s"
NUM_CLASSES = 6
CLASS_NAMES = ['Classical', 'Cottage', 'Gothic', 'Hitech', 'Japanese', 'Minimalism']

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2