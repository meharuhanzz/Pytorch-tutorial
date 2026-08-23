"""PyTorch Day 5 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn

torch.manual_seed(0)

# 1. Using main.py's make_blob-style approach, create a 2-cluster (not 3)
#    2D dataset (100 points per cluster), with centers of your choice.
#    Build a small classifier (nn.Sequential, 2 -> hidden -> 2) and train
#    it for 50 epochs, printing loss every 10 epochs.
# TODO

# 2. Add a train/test split (80/20) to your exercise 1 setup, and report
#    final test accuracy separately from train accuracy.
# TODO

# 3. Try changing the optimizer from Adam to SGD (same model/data as
#    exercise 1) with lr=0.05. Does it converge as fast over the same
#    50 epochs? Print the final loss for both and compare.
# TODO

# 4. Deliberately break the training loop by removing the
#    optimizer.zero_grad() call, train for 20 epochs, and observe what
#    happens to the loss (does it still decrease normally, get worse, or
#    behave strangely?). Put your observation in a comment. Then fix it
#    back and confirm training behaves normally again.
# TODO
