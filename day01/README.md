# PyTorch Day 1 — Tensors

Everything in PyTorch is built on the **tensor** — a multi-dimensional
array, similar to a NumPy array, but with two superpowers NumPy doesn't
have: it can track gradients automatically (Day 2), and it can run on a
GPU.

## Creating tensors

```python
torch.tensor([1, 2, 3])   # from a Python list
torch.zeros(2, 3)          # a 2x3 tensor of zeros
torch.ones(2, 3)           # a 2x3 tensor of ones
torch.rand(2, 3)           # random values in [0, 1)
torch.arange(12)           # 0, 1, 2, ..., 11 -- like Python's range()
```

## Shape, dtype, dimensions

- `.shape` — the size along each dimension (e.g. `torch.Size([3, 4])`)
- `.dtype` — the data type (`torch.float32`, `torch.int64`, etc.)
- `.ndim` — how many dimensions

## Operations: element-wise vs. matrix multiplication

This is the single most important distinction to get right on Day 1:

```python
x * y      # element-wise: multiplies matching positions
x @ y       # matrix multiplication (or torch.matmul(x, y))
```

For `x * y`, both tensors need the same shape (or be "broadcastable" — a
NumPy/PyTorch rule for compatible shapes). For `x @ y`, the *inner*
dimensions must match — a `(2, 3)` tensor times a `(3, 4)` tensor gives a
`(2, 4)` result. Mixing these two up is one of the most common early
PyTorch bugs.

## Indexing, slicing, reshaping

Same rules as Python lists and NumPy arrays — `t[0]` for the first row,
`t[:, 0]` for the first column, `t[1, 2]` for a single element.

`.reshape(...)` changes a tensor's shape without changing its data.
Passing `-1` for one dimension tells PyTorch "figure this size out for
me" — `flat.reshape(2, -1)` on a 12-element tensor gives you a `(2, 6)`
tensor automatically.

## Tensors and NumPy

```python
torch.from_numpy(np_array)   # NumPy -> tensor
tensor.numpy()                 # tensor -> NumPy
```

These share the same underlying memory when on CPU — a nice detail, but
not something to worry about yet.

## Devices — writing GPU-ready code from day one

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)
```

You'll write this exact pattern in almost every script from here on.
Writing code this way from the start means it automatically uses a GPU
when one's available, and just as automatically falls back to CPU when
one isn't — no special-casing needed.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
