"""PyTorch Day 7 -- Exercises. Run make_sample_images.py first if you
haven't already. Fill in the TODOs, then run: python3 exercises.py"""
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sample_images")

# 1. Build a transforms.Compose pipeline that resizes to (48, 48),
#    converts to a tensor, and normalizes with mean=[0.5,0.5,0.5],
#    std=[0.5,0.5,0.5]. Load one sample image (see main.py for how) and
#    print its shape and value range after the pipeline.
# TODO

# 2. Load the sample_images/ directory with ImageFolder using your
#    pipeline from exercise 1. Print how many images and classes it
#    found, and the class_to_idx mapping.
# TODO

# 3. Build an augmentation pipeline using at least 3 different
#    augmentation transforms (RandomHorizontalFlip, RandomRotation,
#    ColorJitter, or others from torchvision.transforms docs). Apply it
#    to the same source image 3 times and confirm the resulting tensors
#    are all different from each other.
# TODO

# 4. Wrap your ImageFolder dataset from exercise 2 in a DataLoader with
#    batch_size=6, shuffle=True. Print the shape of one batch's images
#    and its labels.
# TODO
