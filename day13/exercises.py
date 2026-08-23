"""PyTorch Day 13 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create a float32 tensor of shape (5, 5) filled with random values.
#    Convert it to bfloat16 and print both the original and converted
#    tensor's element_size() in bytes, and their dtypes.
# TODO

# 2. Create a small model (any nn.Sequential of your choice) and move it
#    to `device`. Create a random input batch and move it to `device`
#    too. Run a forward pass INSIDE a torch.autocast(device_type=
#    device.type, dtype=torch.bfloat16) block and print the output's
#    dtype.
# TODO

# 3. In your own words (as a comment), explain why GradScaler is needed
#    for float16 autocast but NOT for bfloat16 autocast. (Hint: think
#    about exponent range vs. mantissa precision -- see main.py section 5.)
# TODO

# 4. Write out (as a comment, pseudocode is fine, doesn't need to
#    actually run without a real GPU + data loader) a complete training
#    loop using float16 autocast + GradScaler, following the pattern
#    from main.py's section 5, but for YOUR model/optimizer/loss_fn
#    names instead of the generic ones shown there.
# TODO
