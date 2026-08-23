"""PyTorch Day 12 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import os

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# 1. Create a TinyNet, save its state_dict to "tiny_weights.pt", then
#    create a SECOND TinyNet instance and load the saved weights into
#    it. Confirm both give identical output on the same random input.
# TODO

# 2. Build a checkpoint dict containing "epoch": 10, the model's
#    state_dict, and a made-up "notes": "halfway through training"
#    string. Save it, then load it back and print all three pieces.
# TODO

# 3. Train TinyNet for 5 manual steps on made-up data (any x, y tensors
#    you like, MSE loss, SGD optimizer), saving a checkpoint (model +
#    optimizer state_dicts + epoch number) after EVERY step, always to
#    the same filename (overwriting each time) -- this is the
#    "checkpoint after every epoch" pattern, as opposed to Day 12's
#    "only save on improvement" pattern.
# TODO

# 4. In a comment, explain: if you trained a model, saved ONLY its
#    state_dict (not a full checkpoint with optimizer state), and later
#    wanted to resume training exactly where you left off -- what would
#    be missing, and what problem might that cause?
# TODO

# Remember to clean up any files you create (os.remove) at the end.
