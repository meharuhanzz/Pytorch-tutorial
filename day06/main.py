"""PyTorch Day 6 -- Datasets & DataLoaders.

Run me with:  python3 main.py
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

# ---- 1. Why not just pass the whole dataset through the model at once? ----
# Day 5 did exactly that (X_train, all 120 examples, in one forward pass).
# That works for 120 tiny 2D points, but real datasets can have millions
# of images -- they simply don't fit in memory or on a GPU all at once.
# The fix is MINI-BATCHING: process the data in small chunks ("batches"),
# updating the model's weights after each one.

# ---- 2. Writing your own Dataset ----
# A custom Dataset needs exactly two methods: __len__ (how many examples
# total) and __getitem__ (given an index, return that one example). This
# is the same __len__/__getitem__ pattern Python lists and dicts use,
# which is why "for x in my_dataset" and my_dataset[5] both just work.
class BlobDataset(Dataset):
    def __init__(self, n_per_class=50):
        def make_blob(center, n, label):
            points = torch.randn(n, 2) * 0.5 + torch.tensor(center)
            labels = torch.full((n,), label, dtype=torch.long)
            return points, labels

        x0, y0 = make_blob((0.0, 0.0), n_per_class, 0)
        x1, y1 = make_blob((4.0, 4.0), n_per_class, 1)
        x2, y2 = make_blob((4.0, -4.0), n_per_class, 2)

        self.X = torch.cat([x0, x1, x2])
        self.Y = torch.cat([y0, y1, y2])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


print("=== A custom Dataset ===")
dataset = BlobDataset()
print(f"len(dataset) = {len(dataset)}")
sample_x, sample_y = dataset[0]     # calls __getitem__(0)
print(f"dataset[0] = {sample_x}, label {sample_y}")

# ---- 3. DataLoader -- batching, shuffling, and iteration, for free ----
# Wrapping a Dataset in a DataLoader gives you an object you can loop
# over, and it handles splitting into batches and (optionally) shuffling
# the order every epoch -- both things you'd otherwise write by hand.
print("\n=== DataLoader ===")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"number of batches: {len(loader)}  (150 examples / batch_size 16, rounded up)")

first_batch_x, first_batch_y = next(iter(loader))
print(f"first batch: X shape {first_batch_x.shape}, Y shape {first_batch_y.shape}")

# ---- 4. The Day 5 training loop, now with real mini-batching ----
print("\n=== Training with batches ===")
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0

    for batch_x, batch_y in loader:   # DataLoader gives you one batch at a time
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (logits.argmax(dim=1) == batch_y).sum().item()

    if epoch % 4 == 0 or epoch == num_epochs - 1:
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / len(dataset)
        print(f"epoch {epoch:2d} | avg loss: {avg_loss:.4f} | accuracy: {accuracy:.4f}")

# ---- 5. Why shuffle=True matters ----
# Without shuffling, the model would see class 0's examples, then class
# 1's, then class 2's, in the SAME order every single epoch -- it could
# pick up on that ordering as a spurious pattern rather than learning the
# actual relationship between input and label. shuffle=True (for
# training data) avoids this. Validation/test loaders are usually left
# unshuffled (shuffle=False), since order doesn't matter there and it
# makes results easier to inspect/compare.
print("\n=== shuffle=True vs shuffle=False ===")
loader_shuffled = DataLoader(dataset, batch_size=150, shuffle=True)
loader_unshuffled = DataLoader(dataset, batch_size=150, shuffle=False)

_, labels_shuffled = next(iter(loader_shuffled))
_, labels_unshuffled = next(iter(loader_unshuffled))
print(f"first 10 labels, shuffled:   {labels_shuffled[:10].tolist()}")
print(f"first 10 labels, unshuffled: {labels_unshuffled[:10].tolist()}")
