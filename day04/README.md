# PyTorch Day 4 — Loss Functions & Optimizers

Two pieces glue autograd (Day 2) and models (Day 3) into something that
actually learns: a **loss function** that measures how wrong the model
is, and an **optimizer** that uses the resulting gradients to improve it.

## Loss functions

**`nn.MSELoss`** (Mean Squared Error) — for regression, predicting a
continuous number:

```python
mse = nn.MSELoss()
loss = mse(predictions, targets)   # mean of (prediction - target)^2
```

**`nn.CrossEntropyLoss`** — for classification, predicting which of N
classes something belongs to. This is the loss used everywhere in this
course's other image-classification projects.

```python
cross_entropy = nn.CrossEntropyLoss()
loss = cross_entropy(logits, true_labels)
```

Important detail: `CrossEntropyLoss` expects **raw scores (logits)**
straight from your model's last layer — not probabilities. It applies
softmax internally. Passing already-softmaxed values in gives wrong
results, a common mistake.

Confident + correct predictions produce low loss; confident + wrong
predictions produce high loss — `main.py` shows both side by side.

## Optimizers

On Day 2 you updated a single weight by hand:

```python
w -= learning_rate * w.grad
```

An **optimizer** does exactly this, automatically, for *every* parameter
in your model:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()   # clear old gradients (Day 2's accumulation gotcha)
loss.backward()          # compute new gradients
optimizer.step()          # update every parameter
```

This three-line sequence — `zero_grad()`, `backward()`, `step()` — is the
heart of every PyTorch training loop you'll ever write. Day 5 wraps it in
an actual loop over multiple epochs.

## SGD vs Adam

- **SGD** — the classic, simple algorithm: `param -= lr * grad`
  (optionally smoothed with momentum).
- **Adam** — adapts the effective learning rate per-parameter
  automatically. Usually converges faster with less manual tuning — the
  default choice for most modern deep learning work, including the
  transfer-learning recipe used throughout this repo's plant-classifier
  projects.

Switching between them is usually just changing one line:

```python
torch.optim.SGD(model.parameters(), lr=0.01)
torch.optim.Adam(model.parameters(), lr=0.01)
```

## Run it

```bash
python3 main.py
```

Watch the final example's loss decrease over 10 training steps as the
model learns the relationship `y = 2x + 1` from data alone.

## Exercises

Open `exercises.py` and work through the four TODOs.
