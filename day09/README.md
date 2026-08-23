# PyTorch Day 9 — Training a CNN End-to-End

Everything from Days 6-8 comes together into one real training run:
`ImageFolder` + `DataLoader` (Day 6-7) feeding a CNN (Day 8) through a
full training loop with validation (Day 5).

**Run `python3 make_shapes.py` once first** — generates a 3-class dataset
(circles/squares/triangles, 60 images each) with randomized position,
size, color, and a noisy background. Unlike Day 7's color-only sample
images, telling these apart genuinely requires learning *shape* — a
useful, honest test of whether the CNN is learning something real.

## random_split — an easier train/val split

Day 5 split data by hand (shuffle, then slice). `torch.utils.data.random_split`
does the same thing for a `Dataset` in one line:

```python
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
```

## The full training script, structurally

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()   # model.train(), loop over train_loader
    val_loss, val_acc = evaluate()                 # model.eval(), loop over val_loader
    print(...)
```

Splitting training and evaluation into their own functions (`train_one_epoch()`,
`evaluate()`) is standard practice once a script grows past a few lines —
it also makes the difference between the two crystal clear: training
computes gradients and updates weights (`optimizer.zero_grad()` /
`.backward()` / `.step()`), evaluation does neither (`torch.no_grad()`,
no optimizer calls at all).

## Reading the output

Watch **both** train and validation accuracy across epochs, not just
train. If training accuracy climbs but validation accuracy stalls or
drops, that's **overfitting** — the model memorizing training examples
rather than learning the general pattern. Day 10 covers this properly;
for now, just get used to looking at both numbers side by side.

## Why this matters beyond toy shapes

This exact structure — `ImageFolder` dataset, `DataLoader`, a CNN,
`CrossEntropyLoss` + `Adam`, a train/val loop — is the same fundamental
loop the medicinal plant classifier project elsewhere in this repository
runs. The only real differences there: a much bigger, ImageNet-pretrained
model instead of training from scratch (Day 11 covers exactly this), and
a harder, real dataset instead of synthetic shapes.

## Run it

```bash
python3 make_shapes.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
