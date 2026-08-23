"""PyTorch Day 1 -- Tensors.

Run me with:  python3 main.py
"""
import torch

# ---- 1. Creating tensors ----
# A tensor is PyTorch's core data structure -- like a NumPy array, but it
# can also track gradients (Day 2) and run on a GPU.
print("=== Creating tensors ===")
a = torch.tensor([1, 2, 3])
print(f"a = {a}, dtype = {a.dtype}")

b = torch.tensor([1.0, 2.0, 3.0])
print(f"b = {b}, dtype = {b.dtype}")

zeros = torch.zeros(2, 3)     # a 2x3 tensor of zeros
ones = torch.ones(2, 3)        # a 2x3 tensor of ones
rand = torch.rand(2, 3)         # random values in [0, 1)
print(f"\nzeros:\n{zeros}")
print(f"ones:\n{ones}")
print(f"rand:\n{rand}")

# ---- 2. Shape, dtype, and dimensions ----
print("\n=== Shape & dtype ===")
matrix = torch.rand(3, 4)
print(f"matrix.shape = {matrix.shape}")   # torch.Size([3, 4])
print(f"matrix.dtype = {matrix.dtype}")
print(f"matrix.ndim = {matrix.ndim}")       # number of dimensions

# ---- 3. Tensor operations ----
print("\n=== Operations ===")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")          # element-wise multiply
print(f"x.dot(y) = {x.dot(y)}")    # dot product: 1*4 + 2*5 + 3*6
print(f"x.mean() = {x.mean()}")
print(f"x.sum() = {x.sum()}")

# Matrix multiplication -- very different from element-wise *
m1 = torch.rand(2, 3)
m2 = torch.rand(3, 4)
result = m1 @ m2                    # or torch.matmul(m1, m2)
print(f"\n(2x3) @ (3x4) -> shape {result.shape}")  # torch.Size([2, 4])

# ---- 4. Indexing and slicing (same rules as Python lists / NumPy) ----
print("\n=== Indexing & slicing ===")
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"t =\n{t}")
print(f"t[0] = {t[0]}")            # first row
print(f"t[:, 0] = {t[:, 0]}")      # first column
print(f"t[1, 2] = {t[1, 2]}")      # single element (row 1, col 2)
print(f"t[0:2, 1:] = \n{t[0:2, 1:]}")

# ---- 5. Reshaping ----
print("\n=== Reshaping ===")
flat = torch.arange(12)            # 0, 1, 2, ..., 11
print(f"flat = {flat}")
reshaped = flat.reshape(3, 4)
print(f"reshaped (3x4) =\n{reshaped}")
reshaped2 = flat.reshape(2, -1)    # -1 means "figure out this dimension for me"
print(f"reshaped (2, -1) =\n{reshaped2}")

# ---- 6. Converting between tensors and NumPy ----
import numpy as np
print("\n=== Tensor <-> NumPy ===")
np_array = np.array([1, 2, 3])
from_numpy = torch.from_numpy(np_array)
print(f"torch.from_numpy({np_array}) = {from_numpy}")

back_to_numpy = from_numpy.numpy()
print(f"back to numpy: {back_to_numpy}, type: {type(back_to_numpy)}")

# ---- 7. GPU tensors (device-agnostic code) ----
# This pattern -- checking availability, then using `device` everywhere --
# is something you'll write in almost every PyTorch script from here on.
print("\n=== Device ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

on_device = torch.rand(2, 2).to(device)
print(f"tensor is on: {on_device.device}")
# If you have a GPU, .to(device) moves the tensor's data there, and any
# operations on it run on the GPU. If not, it silently stays on CPU --
# this is why writing device-agnostic code from the start pays off.
