"""PyTorch Day 15 -- Capstone: A Full Image Classifier Project.

Everything from Days 1-14, combined into one real, complete project:
  - Dataset/DataLoader                             (Day 6)
  - transforms + ImageFolder, separate train/val    (Day 7)
  - a CNN's building blocks, understood from Day 8, but here we use...
  - transfer learning (a pretrained backbone)        (Day 11)
  - Dropout regularization                            (Day 10)
  - a proper train/eval loop                           (Day 5, 9)
  - early stopping + saving only the BEST checkpoint    (Day 12)
  - full evaluation: confusion matrix, per-class acc,    (Day 14)
    precision/recall/F1
  - device-agnostic code                                   (Day 1, 13)

Run make_shapes.py FIRST if you haven't:
    python3 make_shapes.py
    python3 main.py
"""
import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# CONFIG -- the kind of section every real training script has
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "shapes")
CHECKPOINT_PATH = os.path.join(HERE, "best_model.pt")

BATCH_SIZE = 8
NUM_EPOCHS = 15
PATIENCE = 4          # stop early if val loss doesn't improve for this many epochs
DROPOUT_P = 0.3
LR_HEAD = 0.001         # stage 1: new head only
LR_FINETUNE = 0.0001     # stage 2: unfrozen backbone, smaller LR
FINETUNE_START_EPOCH = 5  # unfreeze layer4 after this many epochs

torch.manual_seed(42)

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run 'python3 make_shapes.py' first to generate the shapes dataset.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# DATA
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(DATA_DIR, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Dataset: {len(full_dataset)} images, {num_classes} classes: {class_names}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")


# ============================================================
# MODEL -- pretrained backbone, new head, with dropout
# ============================================================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT_P),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model.to(device)


model = build_model()

# Start frozen except the new head -- feature extraction first (Day 11).
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)


# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================
def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


# ============================================================
# TRAINING LOOP -- staged fine-tuning + early stopping + best-checkpoint
# ============================================================
print("\n=== Training ===")
best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Stage 2: unfreeze the last block partway through, at a smaller LR
    # (Day 11's staged fine-tuning recipe).
    if epoch == FINETUNE_START_EPOCH:
        print(f"--- epoch {epoch}: unfreezing layer4, switching to lr={LR_FINETUNE} ---")
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE
        )

    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(f"epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(best_state)   # restore the BEST checkpoint, not just the last epoch


# ============================================================
# SAVE THE TRAINED MODEL (Day 12)
# ============================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "best_val_loss": best_val_loss,
}, CHECKPOINT_PATH)
print(f"\nSaved best checkpoint to {CHECKPOINT_PATH}")


# ============================================================
# FULL EVALUATION (Day 14)
# ============================================================
print("\n=== Final Evaluation ===")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Final validation accuracy: {accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion matrix (rows=true, cols=predicted), classes={class_names}:")
print(cm)

print("\nPer-class accuracy:")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for name, acc in zip(class_names, per_class_acc):
    print(f"  {name}: {acc:.4f}")

print("\nFull classification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


# ============================================================
# LOADING THE SAVED MODEL BACK (a quick demo, Day 12)
# ============================================================
print("=== Confirming the saved checkpoint loads and predicts correctly ===")
loaded = torch.load(CHECKPOINT_PATH, weights_only=True)
fresh_model = build_model()
fresh_model.load_state_dict(loaded["model_state_dict"])
fresh_model.eval()

sample_images, sample_labels = next(iter(val_loader))
with torch.no_grad():
    original_preds = model(sample_images.to(device)).argmax(dim=1).cpu()
    loaded_preds = fresh_model(sample_images.to(device)).argmax(dim=1).cpu()

print(f"Original model predictions: {original_preds.tolist()}")
print(f"Loaded model predictions:   {loaded_preds.tolist()}")
print(f"Identical: {torch.equal(original_preds, loaded_preds)}")

print("""
=== You've now built a complete PyTorch project ===
Real data pipeline, pretrained transfer learning, regularization, a
proper staged training loop with early stopping, best-checkpoint saving,
full evaluation metrics, and confirmed the saved model reloads correctly.
This is the same shape as the real projects elsewhere in this account --
you're ready to build your own from here.
""")
