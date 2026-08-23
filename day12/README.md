# PyTorch Day 12 — Saving, Loading & Checkpointing

## state_dict — a model's weights as a plain dictionary

```python
model.state_dict()   # OrderedDict: {"layer1.weight": tensor(...), "layer1.bias": tensor(...), ...}
```

Every `nn.Module` can produce this — a mapping from each parameter's name
to its current value. This is the thing you save, not the model object
itself (more on why below).

## Saving and loading weights

```python
torch.save(model.state_dict(), "weights.pt")

new_model = SmallNet()                              # recreate the architecture from your code
new_model.load_state_dict(torch.load("weights.pt", weights_only=True))
```

Notice: the architecture itself isn't saved — only the numbers. Your
Python class definition (`SmallNet`) is what recreates the structure;
loading just fills in the learned values. This means the class definition
you load into must match the one that produced the saved weights (same
layer names, same shapes).

## Full checkpoints — for resuming training

Weights alone are enough to *use* a trained model, but not enough to
*resume training* from where you stopped — you also need the optimizer's
internal state (Adam tracks per-parameter moving averages, for example)
and which epoch you were on:

```python
checkpoint = {
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": 0.234,
}
torch.save(checkpoint, "checkpoint.pt")
```

Loading it back restores all three pieces, so training can continue
exactly where it left off, rather than the optimizer having to "warm up"
its internal state again from scratch.

## Why state_dict, not the whole model object

`torch.save(model, path)` is technically possible — it pickles the
entire Python object, architecture included. This seems convenient (no
need to recreate the class before loading), but it's fragile: if your
model class or a library it depends on changes even slightly later,
loading an old pickled model can fail in confusing, hard-to-debug ways.
**Saving `state_dict` and recreating the architecture from your current
code is the robust, recommended approach**, and what every real project
in this account uses.

## Save-best-only

The pattern used throughout this account's actual training scripts:

```python
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")   # only overwrite on improvement
```

This guarantees the saved file is always the *best-performing* checkpoint
seen so far — not just whatever the training loop happened to produce on
its last epoch, which (especially without early stopping, Day 5's
lesson) might actually be a worse, overfit version.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
