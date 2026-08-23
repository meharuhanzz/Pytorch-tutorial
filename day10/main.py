"""PyTorch Day 10 -- Overfitting & Regularization.

Run make_shapes.py FIRST if you haven't:
    python3 make_shapes.py
    python3 main.py
"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run 'python3 make_shapes.py' first to generate the shapes dataset.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
full_dataset = ImageFolder(DATA_DIR, transform=transform)

# A SMALL training set on purpose -- overfitting shows up fastest and
# most clearly when there isn't much data to learn from.
train_size = 30
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print(f"train: {len(train_dataset)} images, val: {len(val_dataset)} images "
      f"(deliberately tiny train set, to make overfitting show up clearly)")


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ShapeCNN(nn.Module):
    """dropout_p=0 and weight_decay=0 reproduces an unregularized model;
    set them to non-zero to see the difference."""

    def __init__(self, num_classes=3, dropout_p=0.0):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),   # see section 2 below
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_step(model, loader, loss_fn, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_and_report(model, optimizer, num_epochs, label):
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n=== {label} ===")
    for epoch in range(num_epochs):
        train_loss, train_acc = run_step(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_step(model, val_loader, loss_fn)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            gap = train_acc - val_acc
            print(f"epoch {epoch + 1:2d}/{num_epochs} | "
                  f"train acc {train_acc:.3f} | val acc {val_acc:.3f} | "
                  f"train-val gap {gap:+.3f}")
    return train_acc, val_acc


# ---- 1. The overfitting problem, demonstrated ----
# A model with no regularization, trained for a lot of epochs on very
# little data, tends to just MEMORIZE the training examples rather than
# learning the general pattern -- training accuracy climbs toward 100%
# while validation accuracy plateaus or gets worse.
overfit_model = ShapeCNN(dropout_p=0.0).to(device)
overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), lr=0.001)
train_and_report(overfit_model, overfit_optimizer, num_epochs=40,
                  label="No regularization (watch the train-val gap grow)")

# ---- 2. Dropout -- randomly disabling neurons during training ----
# nn.Dropout(p) randomly zeroes out each activation with probability p
# DURING TRAINING ONLY (this is exactly why model.train()/model.eval()
# from Day 3 matters -- Dropout checks that flag and does nothing during
# eval). This forces the network to not over-rely on any single neuron,
# since it might be "switched off" on any given step -- a form of
# built-in redundancy that tends to generalize better.
print("\n=== nn.Dropout demonstration ===")
dropout = nn.Dropout(p=0.5)
sample = torch.ones(10)
dropout.train()
print(f"input:  {sample}")
print(f"after Dropout(0.5), train mode: {dropout(sample)}")   # roughly half zeroed
dropout.eval()
print(f"after Dropout(0.5), eval mode:  {dropout(sample)}")   # unchanged -- disabled

# ---- 3. Weight decay (L2 regularization) -- penalizing large weights ----
# Adding weight_decay to the optimizer nudges every weight slightly
# toward zero on each step, in addition to the normal gradient update.
# This discourages the model from relying on any single weight becoming
# extremely large to fit the training data exactly.
regularized_model = ShapeCNN(dropout_p=0.5).to(device)
regularized_optimizer = torch.optim.Adam(
    regularized_model.parameters(), lr=0.001, weight_decay=1e-3
)
train_and_report(regularized_model, regularized_optimizer, num_epochs=40,
                  label="With Dropout(0.5) + weight_decay=1e-3")

print("""
=== Summary ===
Compare the final train-val gap in the two runs above. Regularization
(Dropout + weight decay) generally keeps validation accuracy closer to
training accuracy, at the cost of training accuracy itself climbing a
little slower/lower -- that tradeoff is usually worth it, since what you
actually care about is how the model performs on data it HASN'T seen.

Other regularization tools you already have from earlier days, without
realizing it:
  - Data augmentation (Day 7)      -- effectively more, varied training data
  - A bigger training set                -- the single most effective fix,
                                             when you can get one
  - Early stopping (used throughout    -- stop training once val loss stops
    this repo's other projects)          improving, rather than training a
                                          fixed number of epochs regardless
""")
