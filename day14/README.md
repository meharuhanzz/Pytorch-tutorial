# PyTorch Day 14 — Evaluation & Metrics

A single accuracy number can hide a lot. Today: the tools for actually
understanding *how* a model is right or wrong, not just *how often*.

## Collecting predictions first

Metrics beyond a running accuracy average need every prediction and every
true label gathered together:

```python
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        preds = model(images).argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
```

## Confusion matrix — which classes get confused with which

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
```

Each row is a true class, each column a predicted class. The diagonal is
correct predictions; everything off-diagonal is a specific *kind* of
mistake — `cm[0][1]` tells you exactly how many true class-0 examples
got wrongly predicted as class 1. This is exactly the tool used to
analyze per-class behavior in this account's real classifier projects
(the exact same 3-way plain-ViT/hybrid/PVT comparison run earlier this
session lives or dies on reading confusion matrices like this one).

## Per-class accuracy

```python
per_class_acc = cm.diagonal() / cm.sum(axis=1)
```

The diagonal divided by each row's total. This is precisely the metric
saved to the `validation_<run>.csv` files in this account's real training
scripts — an overall accuracy of 98% can still hide one specific class
sitting at 40%, and per-class accuracy is how you catch that.

## Precision, recall, F1

- **Precision** — of everything predicted as class X, how much of it
  really was X? (Are we crying wolf too often?)
- **Recall** — of everything that really was class X, how much did we
  actually catch? (Are we missing too many?)
- **F1** — the harmonic mean of the two, a single number balancing both.

```python
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=class_names))
```

These matter most when classes are imbalanced or when false positives
and false negatives have different real-world costs — plain accuracy
alone can look fine while one of these is quietly bad.

## Inspecting individual predictions

Aggregate numbers tell you *that* something's wrong; looking at specific
predictions (especially wrong, confident ones) is often how you discover
*why* — an ambiguous image, a mislabeled example, or a genuinely hard
case worth more training data for.

## Run it

```bash
python3 make_shapes.py   # once (same generator as Day 9/10/11)
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
