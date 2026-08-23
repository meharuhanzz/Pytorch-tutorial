# PyTorch Day 15 — Capstone: A Full Image Classifier Project

Congratulations on making it through 14 days — this project combines
everything into one complete, real training script.

## What it does

Classifies 4 shapes (circle, square, triangle, star) from small
synthetic images, using a pretrained ResNet18 backbone, with a proper
staged fine-tuning schedule, early stopping, best-checkpoint saving, and
a full evaluation report at the end.

## Where each earlier day shows up

| Concept | Day | Where in this project |
|---|---|---|
| Dataset/DataLoader | 6 | `train_loader` / `val_loader` |
| transforms + ImageFolder | 7 | `transform`, `full_dataset` |
| CNN fundamentals | 8 | Understood, though today uses a pretrained one instead of training from scratch |
| A full train/val loop | 5, 9 | `run_epoch()`, the main training loop |
| Overfitting & Dropout | 10 | `nn.Dropout(DROPOUT_P)` in the new head |
| Transfer learning | 11 | `models.resnet18(weights=...)`, staged unfreezing |
| Saving/loading checkpoints | 12 | `best_state`, `torch.save(...)`, the reload-and-confirm section |
| GPU/device-agnostic code | 1, 13 | `device = torch.device(...)` used throughout |
| Evaluation & metrics | 14 | Confusion matrix, per-class accuracy, classification report |

## The staged fine-tuning schedule

```python
if epoch == FINETUNE_START_EPOCH:
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(trainable_params, lr=LR_FINETUNE)  # smaller LR
```

Epochs 0-4 train only the new head (backbone frozen). From epoch 5
onward, `layer4` (ResNet18's last block) unfreezes too, at a smaller
learning rate. This is a two-stage version of the exact recipe used in
this account's real image classifiers — Exercise 3 asks you to extend it
to a full three-stage version.

## Early stopping + best-checkpoint saving, together

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        break

model.load_state_dict(best_state)   # restore the BEST epoch, not the last one
```

Notice the model reloads `best_state` at the end — if validation loss got
worse in the final few epochs before stopping, the saved/returned model
is still the best one seen, not whatever the last epoch happened to
produce.

## Confirming the saved model actually works

The script doesn't just save a checkpoint and trust it — it reloads it
into a fresh model instance and confirms the predictions match, right
there in the same run. This "does the save/load round-trip actually
work" check is cheap insurance worth having in any real project.

## Run it

```bash
python3 make_shapes.py   # once -- note this generates 4 classes now, not 3
python3 main.py
```

## Exercises

Open `exercises.py`. These extend the capstone project directly rather
than starting fresh — exactly the kind of work you'd do on a real
project after getting an initial version working.

## Where to go from here

You've now seen the complete shape of a real PyTorch project. Natural
next steps beyond this course: learning about more advanced architectures
(the medicinal plant classifier elsewhere in this account compares plain
ViT, a CNN+ViT hybrid, and PVT — worth reading through once you're
comfortable here), experiment tracking tools (Weights & Biases,
TensorBoard — used throughout this account's real training scripts),
and larger, real-world datasets instead of synthetic shapes.
