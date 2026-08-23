"""PyTorch Day 10 -- Exercises. Run make_shapes.py first if you haven't.
Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn

torch.manual_seed(0)

# 1. Create nn.Dropout(p=0.3) and a tensor of 20 ones. Run it through the
#    dropout layer in train mode 3 times and print each result -- notice
#    the zeroed positions differ each time (it's random per call).
# TODO

# 2. Create two Adam optimizers for two copies of the same tiny model
#    (nn.Linear(4, 2) is fine): one with weight_decay=0, one with
#    weight_decay=0.1. Run a few manual training steps on made-up data
#    for both and compare how much the weights grow -- does the
#    weight_decay=0.1 version's weights stay smaller?
# TODO

# 3. Using main.py's ShapeCNN and train_and_report pattern, try THREE
#    dropout values (0.0, 0.3, 0.7) with the same weight_decay=0, 30
#    epochs each. Which train-val gap is smallest? Is a very high
#    dropout (0.7) always better, or does it start hurting training
#    accuracy too much?
# TODO

# 4. In your own words (as a comment), explain why Dropout is applied
#    only during training and disabled during evaluation -- what would
#    go wrong if it stayed active at inference time?
# TODO
