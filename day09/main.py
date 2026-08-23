"""PyTorch Day 9 -- Training a CNN Image Classifier End-to-End.

Pulls together Day 6 (Dataset/DataLoader), Day 7 (transforms/ImageFolder),
and Day 8 (CNN architecture) into a complete, real training run.

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
print(f"Using device: {device}")

# ---- 1. Data (Day 6 + Day 7) ----
transform = transforms.Compose([
    transforms.ToTensor(),
])

full_dataset = ImageFolder(DATA_DIR, transform=transform)
print(f"Dataset: {len(full_dataset)} images, classes: {full_dataset.classes}")

# random_split is a convenient shortcut for the manual train/test split
# you wrote by hand on Day 5 -- splits a Dataset into pieces of the sizes
# you ask for.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ---- 2. Model (Day 8) ----
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ShapeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 8),    # 48x48 -> 24x24
            ConvBlock(8, 16),   # 24x24 -> 12x12
            ConvBlock(16, 32),  # 12x12 -> 6x6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = ShapeCNN(num_classes=len(full_dataset.classes)).to(device)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- 3. Loss and optimizer (Day 4) ----
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ---- 4. Training loop, with a validation pass every epoch (Day 5 + Day 6) ----
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


print("\n=== Training ===")
num_epochs = 15
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()
    print(f"epoch {epoch + 1:2d}/{num_epochs} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")

print("\n=== Done ===")
print("A CNN trained from scratch on ~180 tiny images just learned to "
      "tell circles, squares, and triangles apart from raw pixels alone -- "
      "no hand-written 'count the corners' logic anywhere. This is the "
      "same fundamental loop the medicinal plant classifier elsewhere in "
      "this repo runs, just with a bigger, pretrained model (Day 11) and "
      "a harder, real-world dataset.")
