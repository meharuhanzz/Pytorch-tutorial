# PyTorch Day 6 — Datasets & DataLoaders

Day 5 passed an entire 120-example dataset through the model in one
shot. Real datasets can have millions of examples — far too much to fit
in memory or on a GPU at once. The fix is **mini-batching**: process the
data in small chunks, updating weights after each one.

## Writing a custom Dataset

Only two methods required:

```python
class BlobDataset(Dataset):
    def __init__(self, ...):
        # load/generate your data here
        self.X = ...
        self.Y = ...

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
```

- **`__len__`** — how many examples total (`len(dataset)`).
- **`__getitem__`** — given an index, return that one example
  (`dataset[5]`).

This is the same pattern Python's built-in lists and dicts use, which is
why a `Dataset` can be indexed and iterated just like they can.

## DataLoader — batching and shuffling for free

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_x, batch_y in loader:
    ...
```

Wrapping a `Dataset` in a `DataLoader` gives you something you can loop
over directly, automatically splitting the data into batches of the size
you asked for, and (if `shuffle=True`) presenting them in a different
random order each epoch.

## The Day 5 loop, now with real batching

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in loader:   # one batch at a time, not everything at once
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Same four-line core from Day 5 — it's now just nested inside a loop over
batches, which is nested inside a loop over epochs. This nested-loop
shape (epochs → batches → the four training-step lines) is what basically
every real PyTorch training script looks like.

## shuffle=True vs shuffle=False

Shuffle **training** data every epoch (`shuffle=True`) — otherwise the
model always sees examples in the same order and can pick up on that
ordering as a spurious pattern, rather than the real relationship between
input and label. Leave **validation/test** loaders unshuffled
(`shuffle=False`) — order doesn't matter for evaluation, and keeping it
fixed makes results easier to inspect and compare run to run.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
