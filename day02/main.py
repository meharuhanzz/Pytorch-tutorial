"""PyTorch Day 2 -- Autograd (automatic differentiation).

Run me with:  python3 main.py
"""
import torch

# ---- 1. requires_grad -- telling PyTorch to track a tensor ----
# By default, PyTorch doesn't bother tracking gradients (it costs memory
# and compute). You opt in with requires_grad=True on tensors that need
# to be learned/updated -- almost always your model's weights.
print("=== requires_grad ===")
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")

# ---- 2. Building a computation and calling .backward() ----
# y = x^2. PyTorch remembers HOW y was computed from x (a "computation
# graph"), so it can work out dy/dx automatically.
y = x ** 2
print(f"y = x**2 = {y}")

y.backward()   # compute the gradient of y with respect to every tensor
                # that has requires_grad=True and was used to compute y
print(f"dy/dx (should be 2x = 6.0): {x.grad}")

# ---- 3. A slightly bigger example, by hand vs. autograd ----
print("\n=== Bigger example ===")
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
z = a ** 2 + b ** 3   # z = a^2 + b^3

z.backward()
print(f"z = a**2 + b**3 = {z}")
print(f"dz/da (should be 2a = 4.0): {a.grad}")
print(f"dz/db (should be 3b^2 = 27.0): {b.grad}")
# By hand: dz/da = 2a = 2*2 = 4.0, dz/db = 3b^2 = 3*9 = 27.0 -- matches!

# ---- 4. Gradients accumulate -- you must zero them between steps ----
# This is one of the most common early PyTorch bugs. Calling .backward()
# ADDS to .grad rather than replacing it -- if you don't reset it, your
# next gradient computation gets contaminated by the previous one.
print("\n=== Gradient accumulation ===")
w = torch.tensor(5.0, requires_grad=True)

loss1 = w ** 2
loss1.backward()
print(f"after 1st backward(): w.grad = {w.grad}")   # 2w = 10.0

loss2 = w ** 2
loss2.backward()
print(f"after 2nd backward() WITHOUT zeroing: w.grad = {w.grad}")   # 20.0 -- accumulated!

w.grad.zero_()   # reset to zero before the next computation
loss3 = w ** 2
loss3.backward()
print(f"after zeroing and a 3rd backward(): w.grad = {w.grad}")   # back to 10.0

# In a real training loop (Day 5) you call optimizer.zero_grad() at the
# start of every iteration for exactly this reason.

# ---- 5. Turning off gradient tracking ----
# Once you're done training and just want to USE a model (inference), you
# don't need gradients -- tracking them wastes memory. torch.no_grad()
# temporarily disables tracking.
print("\n=== torch.no_grad() ===")
x = torch.tensor(4.0, requires_grad=True)
with torch.no_grad():
    y = x ** 2
    print(f"y = {y}, y.requires_grad = {y.requires_grad}")   # False -- not tracked

# ---- 6. detach() -- getting a plain tensor out of the graph ----
print("\n=== detach() ===")
x = torch.tensor(4.0, requires_grad=True)
y = x ** 2
y_detached = y.detach()
print(f"y.requires_grad = {y.requires_grad}, y_detached.requires_grad = {y_detached.requires_grad}")

# ---- 7. A tiny "manual gradient descent" step -- the core idea behind training ----
print("\n=== One manual gradient descent step ===")
# Goal: minimize (w - 10)^2 -- the minimum is obviously at w = 10.
w = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

for step in range(5):
    loss = (w - 10) ** 2
    loss.backward()

    with torch.no_grad():   # updating w should NOT itself be tracked
        w -= learning_rate * w.grad

    w.grad.zero_()
    print(f"step {step}: w = {w.item():.3f}, loss = {loss.item():.3f}")

# Watch w creep closer to 10 with each step -- this loop, generalized to
# a whole model's worth of weights, IS how neural network training works
# (Day 5 builds the real version of this).
