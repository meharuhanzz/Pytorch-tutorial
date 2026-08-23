"""PyTorch Day 8 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn

torch.manual_seed(0)

# 1. Create an nn.Conv2d that takes a 1-channel (grayscale) image and
#    produces 4 output channels, with a 5x5 kernel and padding=2. Pass
#    a fake grayscale image (1, 1, 28, 28) through it and print the
#    output shape -- confirm height/width stayed 28x28.
# TODO

# 2. Create an nn.MaxPool2d with kernel_size=2, and apply it twice in a
#    row to a (1, 8, 32, 32) fake feature map. Print the shape after
#    each pooling step.
# TODO

# 3. Define a class SmallCNN(nn.Module) using the ConvBlock pattern from
#    main.py, with exactly 2 ConvBlocks: (3 -> 8) then (8 -> 16), for
#    32x32 input images, ending in Flatten + Linear -> 5 classes. Work
#    out what the flattened size should be before writing the final
#    Linear layer (hint: 32x32 through two 2x pooling steps -> 8x8, with
#    16 channels).
# TODO

# 4. Pass a batch of 6 fake 32x32 RGB images through your SmallCNN from
#    exercise 3 and print the output shape and total parameter count.
# TODO
