# PyTorch Day 3 — Building a Model with nn.Module

## nn.Linear — a fully connected layer

```python
layer = nn.Linear(in_features=3, out_features=1)
output = layer(x)   # computes x @ weight.T + bias
```

Creating an `nn.Linear` automatically creates its `weight` and `bias`
tensors, already set up with `requires_grad=True` — you never manually
create the weights yourself.

## The standard pattern: subclassing nn.Module

Virtually every PyTorch model, from a toy 2-layer network to a
state-of-the-art vision transformer, follows this exact shape:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()          # always call this first
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

- **`__init__`** — declare the layers you'll use (this is where weights
  get created).
- **`forward`** — describe how data actually flows through those layers.
  This is where the model's real logic lives.
- Calling `model(x)` runs `model.forward(x)` for you — don't call
  `.forward()` directly.

## Why non-linear activations matter

Stack two `nn.Linear` layers with *nothing* between them, and
mathematically it's equivalent to just one bigger `nn.Linear` layer — no
matter how many layers you add, you'd still only be able to represent
straight-line (linear) relationships. Inserting a non-linear function
like `nn.ReLU()` between layers is what lets a deep stack of layers
represent genuinely complex, curved functions. `ReLU` itself is simple:
negative numbers become 0, positive numbers pass through unchanged.

## Inspecting a model

```python
sum(p.numel() for p in model.parameters())    # total parameter count
model.named_parameters()                        # (name, tensor) for each one
```

Every parameter here already has `requires_grad=True` — this is exactly
what Day 2's autograd machinery will differentiate through once there's
a loss function (Day 4) to compute a gradient *of*.

## nn.Sequential — a shortcut for simple pipelines

When your model is just "run these layers one after another, nothing
fancier," `nn.Sequential` saves you writing an explicit class:

```python
model = nn.Sequential(
    nn.Linear(10, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 2),
)
```

Reach for a full `nn.Module` subclass instead once you need branching,
skip connections, or any logic beyond a straight line of layers.

## train() vs eval() mode

```python
model.train()   # training mode (the default)
model.eval()     # inference mode
```

This doesn't run anything by itself — it's a flag that specific layer
types (Dropout, BatchNorm — met properly on Day 10) check internally to
behave differently during training vs. inference. Forgetting to call
`model.eval()` before evaluating/predicting is a common, subtle bug once
your model includes those layer types.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
