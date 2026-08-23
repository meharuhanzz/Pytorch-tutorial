"""PyTorch Day 3 -- Building a model with nn.Module.

Run me with:  python3 main.py
"""
import torch
from torch import nn

# ---- 1. A single linear layer ----
# nn.Linear(in_features, out_features) is "y = xW^T + b" -- a fully
# connected layer. It creates its own weight and bias tensors internally,
# already with requires_grad=True.
print("=== nn.Linear ===")
layer = nn.Linear(in_features=3, out_features=1)
print(f"layer.weight = {layer.weight}")
print(f"layer.bias = {layer.bias}")

x = torch.tensor([[1.0, 2.0, 3.0]])   # a "batch" of 1 example, 3 features
output = layer(x)                      # calling the layer like a function runs it
print(f"output = {output}")

# ---- 2. Defining your own model with nn.Module ----
# This is the standard pattern for essentially every PyTorch model you'll
# ever write: subclass nn.Module, define your layers in __init__, and
# describe how data flows through them in forward().
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()   # always call this first -- sets up nn.Module's internals
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 3)
        self.activation = nn.ReLU()   # a non-linearity, more on this below

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


print("\n=== A custom nn.Module ===")
model = SimpleNet()
print(model)   # nn.Module has a nice default printout of its structure

sample_input = torch.rand(2, 4)   # batch of 2 examples, 4 features each
output = model(sample_input)       # this calls model.forward() internally
print(f"input shape: {sample_input.shape}, output shape: {output.shape}")

# ---- 3. Why non-linear activations matter ----
# Without a non-linear function between layers, stacking multiple Linear
# layers is mathematically equivalent to just ONE Linear layer -- you'd
# gain nothing from depth. ReLU (and friends like GELU, Sigmoid) is what
# lets a stack of layers represent genuinely more complex functions.
print("\n=== Why activations matter ===")
relu = nn.ReLU()
test_values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
print(f"input:  {test_values}")
print(f"ReLU:   {relu(test_values)}")   # negative values become 0, positive pass through

# ---- 4. Inspecting a model's parameters ----
print("\n=== Parameters ===")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

for name, param in model.named_parameters():
    print(f"  {name}: shape {tuple(param.shape)}, requires_grad={param.requires_grad}")

# ---- 5. A model with more layers, using nn.Sequential for brevity ----
# nn.Sequential is a shortcut for "just run these layers one after
# another" -- useful when your model really is just a straight pipeline,
# with no branching or custom logic in forward().
print("\n=== nn.Sequential ===")
sequential_model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
print(sequential_model)
out = sequential_model(torch.rand(5, 10))
print(f"output shape: {out.shape}")

# ---- 6. train() vs eval() mode ----
# Some layers (like Dropout and BatchNorm, which you'll meet later)
# behave differently during training vs. inference. Calling
# model.train() or model.eval() switches this behaviour -- it doesn't
# run anything by itself, it's a mode flag other layers check.
print("\n=== train() / eval() ===")
print(f"model.training = {model.training}")   # True by default
model.eval()
print(f"after model.eval(): model.training = {model.training}")
model.train()
print(f"after model.train(): model.training = {model.training}")
