"""PyTorch Day 13 -- GPU & Mixed-Precision Training.

Run me with:  python3 main.py

Note: this machine may not have a GPU attached. Everything below still
runs and demonstrates the real code patterns correctly on CPU -- but the
actual SPEED benefit of mixed precision is GPU-specific (it relies on
GPU tensor cores). Run this same script on a CUDA GPU to see real
speedups; here, focus on understanding the pattern.
"""
import torch
from torch import nn

torch.manual_seed(42)

# ---- 1. Device-agnostic code, recap from Day 1 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Device ===\nUsing: {device}")
if device.type == "cpu":
    print("(No GPU detected on this machine -- the code patterns below are "
          "still correct and will run, just without the real speed benefit "
          "mixed precision provides on a GPU.)")

# ---- 2. Why GPUs matter for deep learning ----
# A GPU has thousands of small cores built for exactly the kind of math
# neural networks do constantly -- large matrix multiplications -- run in
# parallel. Moving a model AND its data to the GPU (.to(device)) is what
# lets training use that parallelism.
model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10)).to(device)
data = torch.rand(64, 100).to(device)
print(f"\nmodel and data both on: {device}")

# ---- 3. Float32 vs float16/bfloat16 -- what "precision" means ----
# By default, PyTorch tensors are float32 -- each number stored with 32
# bits of precision. float16 (half precision) and bfloat16 use only 16
# bits: less memory per number, and GPUs with "tensor cores" can do
# float16/bfloat16 matrix multiplication significantly faster than
# float32. The tradeoff: less numeric precision -- usually fine for deep
# learning, since neural networks are fairly tolerant of small numeric
# errors, but not guaranteed for every operation.
print("\n=== Precision sizes ===")
x32 = torch.rand(3, dtype=torch.float32)
x16 = x32.to(torch.float16)
xbf16 = x32.to(torch.bfloat16)
print(f"float32: {x32}, element size: {x32.element_size()} bytes")
print(f"float16: {x16}, element size: {x16.element_size()} bytes")
print(f"bfloat16: {xbf16}, element size: {xbf16.element_size()} bytes")

# ---- 4. Automatic Mixed Precision (AMP) -- torch.autocast ----
# Rather than manually converting every tensor, torch.autocast
# automatically picks float16/bfloat16 for operations that are safe and
# fast in reduced precision, while keeping numerically sensitive
# operations in float32 -- the "mixed" in mixed precision.
print("\n=== torch.autocast ===")
loss_fn = nn.CrossEntropyLoss()
labels = torch.randint(0, 10, (64,)).to(device)

# On CUDA GPUs, you'd typically autocast to float16 (needs GradScaler,
# see below). bfloat16 works on both CPU and GPU without GradScaler,
# since it has the same exponent range as float32 (less prone to the
# underflow problem float16 can hit) -- used here so this runs correctly
# on CPU too.
autocast_dtype = torch.bfloat16
with torch.autocast(device_type=device.type, dtype=autocast_dtype):
    predictions = model(data)
    loss = loss_fn(predictions, labels)
    print(f"predictions.dtype inside autocast: {predictions.dtype}")

print(f"loss (computed in float32 for stability): {loss.dtype}")
# Notice: the Linear layers' matmuls ran in bfloat16, but the loss
# computation was automatically kept in float32 -- autocast decides this
# per-operation, based on which operations are known to be safe to run
# in reduced precision.

# ---- 5. GradScaler -- needed for float16 (not bfloat16) on CUDA ----
# float16 has a much smaller exponent range than float32, so gradients
# can sometimes underflow to zero during backward(), silently stalling
# training. GradScaler works around this by temporarily scaling the loss
# up before backward(), then scaling gradients back down before the
# optimizer step. This is ONLY needed for float16 -- bfloat16 (used
# above) doesn't have this problem, since it keeps float32's exponent
# range, just with less precision in the mantissa.
print("\n=== The full AMP training step (GPU, float16) ===")
print("""
scaler = torch.amp.GradScaler()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(images)
        loss = loss_fn(outputs, labels)

    scaler.scale(loss).backward()   # scale up before backward
    scaler.step(optimizer)           # unscales gradients, then steps
    scaler.update()                    # adjusts the scale factor for next time
""")

# ---- 6. Why bother? ----
print("=== Why mixed precision is worth using ===")
print("""
On a GPU with tensor cores (most modern NVIDIA GPUs), mixed precision
training typically gives:
  - Faster training -- often 1.5-3x, since float16/bfloat16 matrix
    multiplication runs faster than float32 on tensor cores.
  - Lower memory usage -- roughly half, for the tensors held in reduced
    precision -- letting you use a larger batch size or a bigger model
    in the same GPU memory.
  - Usually little to no accuracy cost, since deep learning tolerates
    the reduced numeric precision well in practice.

It's essentially "free" performance on supported hardware -- which is
why torch.autocast + GradScaler (or bfloat16 autocast without GradScaler)
shows up in most modern, performance-conscious PyTorch training scripts.
""")
