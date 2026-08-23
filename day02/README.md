# PyTorch Day 2 — Autograd

Autograd is the feature that makes PyTorch a *deep learning* framework
rather than just "NumPy with tensors" — it automatically computes
derivatives for you, which is exactly what training a neural network
needs (backpropagation is just repeated derivative computation).

## requires_grad

```python
x = torch.tensor(3.0, requires_grad=True)
```

This tells PyTorch: "remember every operation done to `x`, so I can later
ask for the derivative of some result with respect to `x`." You'll set
this on your model's learnable weights — not on your input data.

## .backward()

```python
y = x ** 2
y.backward()
print(x.grad)   # dy/dx = 2x -- evaluated at x's current value
```

`.backward()` walks backward through every operation that produced `y`
(the "computation graph") and fills in `.grad` on every tensor along the
way that has `requires_grad=True`. This is automatic — you never write
the derivative formula yourself.

## Gradients accumulate — the #1 early gotcha

Calling `.backward()` **adds** to `.grad` rather than replacing it:

```python
loss1.backward()   # w.grad = 10.0
loss2.backward()   # w.grad = 20.0 -- accumulated, not replaced!
```

If you don't reset (`w.grad.zero_()`) between training steps, gradients
from previous steps silently corrupt your current one. In a real training
loop (Day 5), this is why `optimizer.zero_grad()` is the very first line
inside the loop, every single iteration.

## torch.no_grad() and .detach()

Once you're done training, you don't need gradient tracking anymore —
it just wastes memory during inference:

```python
with torch.no_grad():
    predictions = model(input_data)   # no graph built, faster & lighter
```

`.detach()` does something similar for a single tensor — it gives you a
copy that's no longer connected to the computation graph.

## Manual gradient descent — the core training idea, in miniature

```python
w = torch.tensor(0.0, requires_grad=True)
for step in range(5):
    loss = (w - 10) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()
```

Every neural network training loop is a generalization of exactly this:
compute a loss, call `.backward()`, nudge every weight a little in the
direction that reduces the loss, reset the gradients, repeat. Day 5 turns
this into the real thing using `nn.Module` (Day 3) and an optimizer
(Day 4) instead of doing the weight update by hand.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
