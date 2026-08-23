"""PyTorch Day 6 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# 1. Write a custom Dataset class `NumbersDataset` that, given a list of
#    numbers in __init__, returns (number, number ** 2) from
#    __getitem__ (i.e. it's a tiny "predict the square" dataset). Create
#    one with numbers 1 through 20 and print its length and item [5].
# TODO

# 2. Wrap your NumbersDataset in a DataLoader with batch_size=4,
#    shuffle=True. Loop over it once and print each batch's contents.
# TODO

# 3. Using the BlobDataset pattern from main.py, create a dataset with
#    n_per_class=200 (so 600 total points) and a DataLoader with
#    batch_size=32. Print how many batches that produces per epoch.
# TODO

# 4. Train the same model/loop as main.py's "Training with batches"
#    section, but using batch_size=8 instead of 16. Does it take a
#    different number of batches per epoch? Does final accuracy after
#    20 epochs look similar?
# TODO
