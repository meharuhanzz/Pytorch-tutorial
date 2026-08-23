# PyTorch Day 13 — GPU & Mixed-Precision Training

**Note on this lesson:** the actual *speedup* from mixed precision is
GPU-specific (it relies on tensor cores). If you're running this on a
machine without a GPU, the code patterns below are still 100% correct
and will run — you just won't see the real performance benefit until you
run the same code on CUDA hardware. Focus on understanding the pattern.

## Device-agnostic code (recap)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

Same pattern from Day 1 — write it this way and your code automatically
uses a GPU when one's available.

## Why GPUs matter here

A GPU has thousands of small cores built for exactly the kind of math
neural networks do constantly: large matrix multiplications, run in
parallel. That's the whole reason `.to(device)` matters — it's what
actually engages that parallel hardware.

## Precision: float32 vs float16/bfloat16

PyTorch tensors default to `float32` — 32 bits of precision per number.
`float16` and `bfloat16` use only 16 bits: half the memory, and GPUs with
tensor cores can multiply matrices in these formats significantly faster
than in `float32`. The tradeoff is reduced numeric precision — usually
fine for deep learning, since neural networks tend to tolerate small
numeric errors well.

## torch.autocast — automatic mixed precision

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    predictions = model(data)
    loss = loss_fn(predictions, labels)
```

Rather than manually converting every tensor, `autocast` automatically
picks reduced precision for operations that are safe and fast that way,
while keeping numerically sensitive operations (like the loss
computation) in `float32`. This per-operation decision-making is the
"mixed" in mixed precision.

## GradScaler — needed for float16, not bfloat16

`float16` has a much smaller exponent range than `float32`, so gradients
can sometimes underflow to zero during `.backward()`, silently stalling
training. `GradScaler` works around this by temporarily scaling the loss
up before the backward pass, then scaling gradients back down before the
optimizer step:

```python
scaler = torch.amp.GradScaler()

with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(images)
    loss = loss_fn(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

`bfloat16` keeps `float32`'s exponent range (just with less mantissa
precision), so it doesn't have this underflow problem — no `GradScaler`
needed, which is why `main.py`'s demo uses `bfloat16` to run correctly
on CPU as well as GPU.

## Why bother

On supported GPU hardware, mixed precision typically gives faster
training (often 1.5-3x) and roughly half the memory usage for the
tensors held in reduced precision — usually with little to no accuracy
cost. It's close to "free" performance, which is why this pattern shows
up in most modern, performance-conscious PyTorch training scripts.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
