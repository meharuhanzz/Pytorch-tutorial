"""PyTorch Day 5 -- The Training Loop.

Putting Days 1-4 together: tensors, autograd, nn.Module, loss functions,
and optimizers, combined into the complete pattern you'll reuse for
every model you ever train in PyTorch.

Run me with:  python3 main.py
"""
import torch
from torch import nn

torch.manual_seed(42)   # reproducible "random" data/weights for this demo

# ---- 1. Make a toy classification dataset ----
# Three clusters of 2D points, each cluster is one class. In a real
# project this section is replaced by a Dataset/DataLoader (Day 6) --
# but the TRAINING LOOP shape below is identical either way.
print("=== Toy dataset: 3 clusters in 2D ===")


def make_blob(center, n_points, label):
    points = torch.randn(n_points, 2) * 0.5 + torch.tensor(center)
    labels = torch.full((n_points,), label, dtype=torch.long)
    return points, labels


x0, y0 = make_blob((0.0, 0.0), 50, 0)
x1, y1 = make_blob((4.0, 4.0), 50, 1)
x2, y2 = make_blob((4.0, -4.0), 50, 2)

X = torch.cat([x0, x1, x2])   # shape: (150, 2)
Y = torch.cat([y0, y1, y2])   # shape: (150,)
print(f"X.shape = {X.shape}, Y.shape = {Y.shape}")
print(f"Classes present: {Y.unique().tolist()}")

# ---- 2. Split into train and test sets ----
# A quick manual split here -- proper train/val practice is Day 10.
# Shuffle first so each split has a mix of all three classes.
perm = torch.randperm(len(X))
X, Y = X[perm], Y[perm]

split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]
print(f"\ntrain size: {len(X_train)}, test size: {len(X_test)}")

# ---- 3. Define the model (Day 3) ----
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3),   # 3 output logits, one per class
)

# ---- 4. Loss function and optimizer (Day 4) ----
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# ---- 5. THE TRAINING LOOP -- this exact shape recurs everywhere ----
print("\n=== Training ===")
num_epochs = 100

for epoch in range(num_epochs):
    model.train()   # training mode (Day 3) -- matters once Dropout/BatchNorm are in play

    # forward pass
    logits = model(X_train)
    loss = loss_fn(logits, Y_train)

    # backward pass + weight update (Day 2 + Day 4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        predictions = logits.argmax(dim=1)
        train_acc = (predictions == Y_train).float().mean()
        print(f"epoch {epoch:3d} | loss: {loss.item():.4f} | train acc: {train_acc.item():.4f}")

# ---- 6. Evaluating on held-out data ----
print("\n=== Evaluation on test set ===")
model.eval()   # inference mode
with torch.no_grad():   # no need to track gradients for evaluation (Day 2)
    test_logits = model(X_test)
    test_predictions = test_logits.argmax(dim=1)
    test_acc = (test_predictions == Y_test).float().mean()

print(f"Test accuracy: {test_acc.item():.4f}")
print(f"Predictions:  {test_predictions.tolist()}")
print(f"True labels:  {Y_test.tolist()}")

# ---- 7. The general shape, summarized ----
print("""
=== The pattern you just saw, generalized ===
for epoch in range(num_epochs):
    model.train()
    for batch in data:                 # Day 6 replaces the whole-dataset
        predictions = model(batch_x)    # pass here with mini-batches
        loss = loss_fn(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Every PyTorch project in this course -- and every one you write after
it -- is a variation on this same loop.
""")
