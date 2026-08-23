"""PyTorch Day 9 -- Exercises. Run make_shapes.py first if you haven't.
Fill in the TODOs, then run: python3 exercises.py"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

# 1. Load the shapes dataset with ImageFolder + ToTensor, split it
#    90/10 instead of 80/20 using random_split, and print the sizes of
#    both splits.
# TODO

# 2. Modify the ShapeCNN from main.py (copy the class in here) to use
#    only 2 ConvBlocks instead of 3 (3 -> 8 -> 16, no third block).
#    Work out the new flattened size by hand (48x48 through two 2x
#    poolings -> 12x12, 16 channels) before writing the Linear layer.
# TODO

# 3. Train your 2-block model from exercise 2 for 15 epochs using the
#    same loop structure as main.py. Does it reach similar accuracy to
#    the 3-block version, faster, slower, or does it plateau lower?
# TODO

# 4. Try changing the optimizer's learning rate from 0.001 to 0.01 (keep
#    the original 3-block ShapeCNN). Train for 15 epochs and compare the
#    loss curve to main.py's -- does a higher LR converge faster, or
#    does it get unstable (loss jumping around instead of decreasing
#    smoothly)?
# TODO
