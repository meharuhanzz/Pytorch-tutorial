"""PyTorch Day 11 -- Exercises. Run make_shapes.py first if you haven't.
Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn
from torchvision import models

# 1. Load a pretrained resnet18 (see main.py). Print its top-level
#    children (hint: model.named_children()) to see the overall
#    structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc.
# TODO

# 2. Replace the final layer for a 5-class problem instead of 3. Confirm
#    the new model.fc's out_features is 5.
# TODO

# 3. Freeze everything except model.fc AND model.layer4 (skip the
#    intermediate Stage 1 head-only step from main.py -- unfreeze both
#    at once here). Print the trainable parameter count and percentage,
#    same style as main.py.
# TODO

# 4. Load a DIFFERENT pretrained model -- models.mobilenet_v2(weights=...)
#    instead of resnet18. Print its total parameter count and compare to
#    resnet18's ~11.7M. (Note: mobilenet_v2's final layer is at
#    model.classifier[1], not model.fc -- different architectures name
#    their final layer differently, worth knowing when you explore new
#    pretrained models.)
# TODO
