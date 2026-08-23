"""PyTorch Day 11 -- Transfer Learning.

This is the technique used throughout the "real" projects elsewhere in
this account's work -- training a huge model from scratch needs enormous
amounts of data and compute. Transfer learning sidesteps that: start from
a model already trained on millions of images (ImageNet), and adapt it
to your much smaller, specific task.

Run make_shapes.py FIRST if you haven't:
    python3 make_shapes.py
    python3 main.py
"""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run 'python3 make_shapes.py' first to generate the shapes dataset.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. Load a pretrained model ----
# resnet18 here was already trained on ImageNet (1.2 million images,
# 1000 classes) -- it already knows how to detect edges, textures,
# shapes, and combinations of those, from all that prior training.
print("=== Loading a pretrained model ===")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(f"Loaded resnet18, {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Original final layer: {model.fc}")   # Linear(512, 1000) -- 1000 ImageNet classes

# ---- 2. Replace the final layer for YOUR task ----
# The 1000-class ImageNet head is useless for a 3-class shapes problem --
# swap it for a fresh, randomly-initialized layer of the right size.
# Everything BEFORE this layer (all the learned edge/texture/shape
# detectors) stays exactly as pretrained.
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)
print(f"New final layer: {model.fc}")
model = model.to(device)

# ---- 3. Freezing -- "feature extraction" mode ----
# Freezing every layer except the new head means: use the pretrained
# model purely as a fixed feature extractor, only training the new
# classifier on top. Fast, and works surprisingly well when your data is
# similar in kind to ImageNet (real-world photos).
print("\n=== Freezing everything except the new head ===")
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} parameters "
      f"({100 * trainable / total:.2f}%)")

# ---- 4. Data (resized to 224x224 -- what resnet18 was trained on) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(DATA_DIR, transform=transform)
train_size = 40
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print(f"\ntrain: {len(train_dataset)}, val: {len(val_dataset)}")


def run_epoch(model, loader, loss_fn, optimizer=None):
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


# ---- 5. Stage 1: train only the new head ----
# Notice how FEW epochs this needs, and how quickly accuracy climbs,
# compared to Day 9/10's from-scratch CNN on the same kind of data --
# this is the entire point of transfer learning. The pretrained features
# are already doing most of the work.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

print("\n=== Stage 1: training only the new head ===")
for epoch in range(5):
    train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
    print(f"epoch {epoch + 1}/5 | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

# ---- 6. Stage 2: unfreeze more layers, fine-tune at a LOWER learning rate ----
# This is exactly the "3-stage fine-tuning" pattern used throughout this
# account's real image-classification projects: start with everything
# frozen except the head, then progressively unfreeze more of the
# network, using a smaller learning rate each time you unfreeze more
# (since those layers already have good, pretrained weights -- you want
# to nudge them gently, not overwrite them with big updates).
print("\n=== Stage 2: unfreezing the last block, fine-tuning at a lower LR ===")
for param in model.layer4.parameters():   # resnet18's last residual block
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable now: {trainable:,} / {total:,} parameters")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001   # smaller LR
)
for epoch in range(5):
    train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
    print(f"epoch {epoch + 1}/5 | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

print("""
=== Summary ===
Two-stage recipe used just now:
  1. Freeze the pretrained backbone, train only a new head -- fast,
     low risk of damaging the pretrained features.
  2. Unfreeze the last block, fine-tune everything trainable at a
     SMALLER learning rate -- lets the model adapt its highest-level
     features to your specific task, without a large update wiping out
     what pretraining already learned.

The real projects elsewhere in this account extend this to a full
3-stage version (head only -> last few blocks -> entire model, with the
learning rate shrinking at each stage) -- same idea, one more step.
""")
