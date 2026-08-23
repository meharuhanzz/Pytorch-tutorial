"""PyTorch Day 4 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn

# 1. Compute the MSE loss between predictions = [1.0, 2.0, 3.0, 4.0] and
#    targets = [1.5, 2.5, 2.5, 4.5]. Work out the answer by hand first
#    (mean of squared differences) as a comment, then confirm with
#    nn.MSELoss().
# TODO

# 2. Create logits for 3 examples across 4 classes (any numbers you
#    like) and true_labels for each. Compute the CrossEntropyLoss.
#    Then change one example's logits to be very confident and CORRECT,
#    and observe the loss drop when you recompute.
# TODO

# 3. Create a single nn.Linear(2, 1) model and an SGD optimizer with
#    lr=0.05. Run ONE manual training step (forward pass, compute MSE
#    loss against a target of your choice, zero_grad, backward, step)
#    and print the weight before and after to confirm it changed.
# TODO

# 4. Extend the "few steps of real training" example from main.py:
#    change the true relationship to y = -3x + 7, run for 30 epochs
#    instead of 10, and print the loss every 5 epochs. Does it get
#    closer to the true w=-3, b=7 than the 10-epoch version did to its
#    target?
# TODO
