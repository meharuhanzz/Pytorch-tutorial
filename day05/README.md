# PyTorch Day 5 — The Training Loop

Everything from Days 1-4 comes together today into the complete pattern
you'll reuse for every model you ever train.

## The full loop

```python
for epoch in range(num_epochs):
    model.train()

    logits = model(X_train)          # forward pass
    loss = loss_fn(logits, Y_train)   # how wrong are we?

    optimizer.zero_grad()              # clear old gradients (Day 2)
    loss.backward()                     # compute new gradients (Day 2)
    optimizer.step()                     # update weights (Day 4)
```

That's it — four lines inside the loop is the entire "learning"
mechanism. Everything else in deep learning (bigger models, more data,
fancier architectures) builds on top of this exact same loop.

## model.train() vs model.eval() — and why it matters here

```python
model.train()   # before training steps
model.eval()     # before evaluation
with torch.no_grad():   # no gradient tracking needed for evaluation
    predictions = model(X_test)
```

You met these individually on Day 2 and Day 3 — this is where they come
together in practice: always evaluate in `eval()` mode, wrapped in
`no_grad()`, since you're not updating weights and don't need the
overhead of gradient tracking.

## Train/test split

`main.py` does a simple shuffle-then-slice split (80% train, 20% test) so
you can check whether the model actually *generalizes* to data it never
trained on, rather than just memorizing the training examples. This is
a preview — Day 10 covers overfitting and proper validation practice in
depth.

## Reading the output

Watch the loss decrease and training accuracy increase over the 100
epochs — that's the model's weights being nudged, step by step, toward
values that separate the three clusters correctly. The final test
accuracy tells you whether it learned the actual *pattern* (cluster
membership) rather than just the specific training points.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs. Exercise 4 is worth
taking seriously — deliberately breaking `zero_grad()` and watching what
happens is the best way to make Day 2's "gradients accumulate" warning
stick.
