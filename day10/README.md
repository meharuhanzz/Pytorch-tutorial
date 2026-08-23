# PyTorch Day 10 — Overfitting & Regularization

## What overfitting looks like

A model **overfits** when it starts memorizing the specific training
examples instead of learning the general pattern behind them. The
telltale sign: **training accuracy keeps climbing while validation
accuracy plateaus or gets worse** — a growing gap between the two.

`main.py` deliberately trains an unregularized model on a tiny training
set (30 images) for 40 epochs to make this gap show up clearly — watch
the `train-val gap` column grow over time.

## nn.Dropout — random redundancy

```python
nn.Dropout(p=0.5)
```

During **training**, Dropout randomly zeroes out each activation with
probability `p`. This forces the network not to over-rely on any single
neuron — since that neuron might be "switched off" on any given step, the
network is pushed toward more redundant, robust representations.

**Critically, Dropout only does this during training.** During evaluation
(`model.eval()`), it does nothing at all — every activation passes
through unchanged. This is exactly why the `model.train()`/`model.eval()`
distinction from Day 3 matters in practice: Dropout literally checks that
flag to decide how to behave.

## weight_decay — penalizing large weights

```python
torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
```

Adding `weight_decay` to an optimizer nudges every weight slightly toward
zero on each update step, on top of the normal gradient-based update.
This discourages the model from relying on any single weight growing
extremely large to fit the training data exactly — another form of
"don't over-commit to what you've memorized."

## Other regularization you already have

- **Data augmentation** (Day 7) — effectively gives the model more,
  varied training examples to learn from, rather than the same fixed set
  repeated every epoch.
- **A bigger training set** — often the single most effective fix, when
  you're able to get one.
- **Early stopping** — used throughout this account's other real
  projects — stop training once validation loss stops improving, rather
  than training a fixed number of epochs regardless of what's happening.

## The tradeoff

Regularization usually makes training accuracy climb a little slower or
plateau a little lower — that's expected, and usually worth it, because
what you actually care about is performance on data the model **hasn't**
seen, which is what validation accuracy measures.

## Run it

```bash
python3 make_shapes.py   # once (same generator as Day 9)
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
