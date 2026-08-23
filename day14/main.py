"""PyTorch Day 14 -- Evaluation & Metrics.

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
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run 'python3 make_shapes.py' first to generate the shapes dataset.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Quickly train a small CNN, same as Day 9, just to have a real
# model to evaluate below. The focus today is what comes AFTER training.
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = ImageFolder(DATA_DIR, transform=transform)
class_names = full_dataset.classes

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(ConvBlock(3, 8), ConvBlock(8, 16), ConvBlock(16, 32))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


print("=== Quick training run (see Day 9 for details) ===")
model = ShapeCNN(len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

print("Training done. Now evaluating.\n")

# ---- 1. Collecting predictions for the whole validation set ----
# Metrics beyond plain accuracy need every prediction and every true
# label collected together, not just a running average -- so we gather
# them all first.
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# ---- 2. Overall accuracy (recap) ----
correct = sum(p == l for p, l in zip(all_preds, all_labels))
accuracy = correct / len(all_labels)
print(f"=== Overall accuracy: {accuracy:.4f} ({correct}/{len(all_labels)}) ===")

# ---- 3. Confusion matrix -- WHICH classes get confused with which ----
# A single accuracy number hides a lot. The confusion matrix shows, for
# every true class, how many examples got predicted as each possible
# class -- the diagonal is correct predictions, everything off-diagonal
# is a specific kind of mistake. This is exactly the tool used to
# analyze per-class behaviour in this account's real classifier projects.
print("\n=== Confusion matrix ===")
cm = confusion_matrix(all_labels, all_preds)
print(f"Classes (in order): {class_names}")
print(cm)
print("\nRow = true class, Column = predicted class.")
print("Reading example: cm[0][1] = number of TRUE 'circle' examples")
print("that were WRONGLY predicted as class index 1.")

# ---- 4. Per-class accuracy ----
# The diagonal of the confusion matrix, divided by each row's total --
# this is exactly the metric saved to the validation_<run>.csv files in
# this account's real training scripts, to spot which specific classes
# a model struggles with (rather than just an overall average that can
# hide a badly-performing class among many good ones).
print("\n=== Per-class accuracy ===")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for name, acc in zip(class_names, per_class_acc):
    print(f"  {name}: {acc:.4f}")

# ---- 5. Precision, recall, F1 -- a fuller picture per class ----
# Accuracy alone can be misleading, especially with imbalanced classes.
#   Precision: of everything predicted as class X, how much really was X?
#   Recall:    of everything that really was class X, how much got caught?
#   F1:        the harmonic mean of precision and recall, a single
#              number balancing both.
print("\n=== Precision / Recall / F1 (via sklearn) ===")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ---- 6. Looking at individual predictions ----
# Beyond aggregate numbers, actually inspecting specific predictions
# (especially wrong ones) is often how you discover WHY a model is
# struggling with a particular class -- ambiguous images, mislabeled
# data, or a genuinely hard case.
print("=== A few individual predictions ===")
model.eval()
sample_images, sample_labels = next(iter(val_loader))
with torch.no_grad():
    logits = model(sample_images.to(device))
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1).cpu()

for i in range(min(5, len(sample_labels))):
    true_name = class_names[sample_labels[i]]
    pred_name = class_names[predictions[i]]
    confidence = probabilities[i][predictions[i]].item()
    correct_mark = "OK" if predictions[i] == sample_labels[i] else "WRONG"
    print(f"  [{correct_mark}] true={true_name}, predicted={pred_name} (confidence {confidence:.2%})")
