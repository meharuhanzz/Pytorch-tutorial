"""PyTorch Day 4 -- Loss Functions & Optimizers.

Run me with:  python3 main.py
"""
import torch
from torch import nn

# ---- 1. What a loss function does ----
# A loss function measures "how wrong" a model's prediction is, as a
# single number. Training a model just means adjusting its weights to
# make this number as small as possible.

# ---- 2. MSELoss -- for regression (predicting a continuous number) ----
print("=== MSELoss (regression) ===")
mse = nn.MSELoss()
predictions = torch.tensor([3.0, 5.0, 2.5])
targets = torch.tensor([3.5, 4.0, 3.0])
loss = mse(predictions, targets)
print(f"predictions: {predictions}")
print(f"targets:     {targets}")
print(f"MSE loss: {loss}")
# By hand: mean of (0.5^2, 1.0^2, 0.5^2) = mean(0.25, 1.0, 0.25) = 0.5

# ---- 3. CrossEntropyLoss -- for classification ----
# This is the one you've actually been using all along in this course's
# other projects (it's the standard loss for "which of N classes is
# this?" problems, like the medicinal plant classifier). It expects RAW
# scores (logits), not probabilities -- it applies softmax internally.
print("\n=== CrossEntropyLoss (classification) ===")
cross_entropy = nn.CrossEntropyLoss()

# 2 examples, 3 possible classes each -- these are raw model outputs
logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1]])
true_labels = torch.tensor([0, 1])   # example 0 is class 0, example 1 is class 1

loss = cross_entropy(logits, true_labels)
print(f"logits:\n{logits}")
print(f"true_labels: {true_labels}")
print(f"CrossEntropy loss: {loss}")

# A confident, CORRECT prediction gives a low loss:
confident_correct = torch.tensor([[10.0, 0.0, 0.0]])
print(f"confident & correct -> loss: {cross_entropy(confident_correct, torch.tensor([0])):.4f}")

# A confident, WRONG prediction gives a high loss:
confident_wrong = torch.tensor([[10.0, 0.0, 0.0]])
print(f"confident & wrong -> loss: {cross_entropy(confident_wrong, torch.tensor([1])):.4f}")

# ---- 4. Optimizers -- automating the "nudge the weights" step from Day 2 ----
# On Day 2 you updated a weight by hand: w -= learning_rate * w.grad
# An optimizer does exactly this, but for every parameter in your model
# at once, and takes care of the torch.no_grad() / zero_grad() bookkeeping
# via its own methods.
print("\n=== Optimizers ===")

model = nn.Linear(3, 1)
print(f"weight before: {model.weight.data}")

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([[10.0]])

prediction = model(x)
loss = nn.MSELoss()(prediction, target)

optimizer.zero_grad()   # step 1: clear old gradients (Day 2's accumulation gotcha!)
loss.backward()          # step 2: compute new gradients
optimizer.step()          # step 3: update every parameter using its gradient

print(f"weight after one step: {model.weight.data}")

# ---- 5. SGD vs Adam -- the two you'll use almost exclusively ----
# SGD (Stochastic Gradient Descent): the simple, classic version -- just
# `param -= lr * grad`, optionally with "momentum" to smooth things out.
# Adam: adapts the effective learning rate per-parameter automatically.
# Usually converges faster with less tuning -- the default choice for
# most modern deep learning, including everything in this course's other
# projects.
print("\n=== SGD vs Adam ===")
model_sgd = nn.Linear(3, 1)
model_adam = nn.Linear(3, 1)

optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)

print(f"SGD optimizer: {optimizer_sgd}")
print(f"Adam optimizer: {optimizer_adam}")
# Both are used identically in a training loop (Day 5) -- swapping
# between them is usually just changing this one line.

# ---- 6. A few epochs of manual training, using everything above ----
print("\n=== A few steps of real training ===")
model = nn.Linear(1, 1)   # learn y = w*x + b for some target w, b
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0], [9.0]])   # true relationship: y = 2x + 1

for epoch in range(10):
    predictions = model(x)
    loss = loss_fn(predictions, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f"epoch {epoch}: loss = {loss.item():.4f}")

learned_w = model.weight.item()
learned_b = model.bias.item()
print(f"\nLearned: y = {learned_w:.2f}*x + {learned_b:.2f}  (target was y = 2*x + 1)")
