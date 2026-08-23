# PyTorch Day 11 — Transfer Learning

This is the technique used throughout this account's real
image-classification projects. Training a large model from scratch needs
enormous amounts of data and compute — the medicinal plant classifier
elsewhere in this account's work has ~65,000 training images and it still
starts from a pretrained model, not from scratch. **Transfer learning**
sidesteps the data/compute problem: start from a model already trained on
millions of images, and adapt it to your specific, much smaller task.

## Loading a pretrained model

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

This model was already trained on ImageNet (1.2 million images, 1000
classes) — it already knows how to detect edges, textures, and shapes,
purely from that prior training.

## Replacing the final layer

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

The original final layer outputs 1000 scores (one per ImageNet class) —
useless for a different task. Swap it for a fresh, randomly-initialized
layer with the right number of outputs for *your* problem. Everything
**before** this layer — all the pretrained edge/texture/shape detectors —
stays exactly as it was.

## Freezing — feature extraction mode

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

Freezing every layer except the new head means the pretrained model is
used purely as a fixed feature extractor — only the new classifier gets
trained. This is fast and works well when your images are broadly similar
in kind to ImageNet (real-world photos of physical objects).

## Why transfer learning needs so few epochs

`main.py`'s Stage 1 trains for just 5 epochs and already reaches solid
accuracy — compare that to Day 9/10's from-scratch CNN on similar-sized
data, which needed far more epochs for far less accuracy. The pretrained
features are already doing most of the work; the new head just needs to
learn how to combine them for your specific classes.

## Staged fine-tuning — unfreezing progressively

```python
for param in model.layer4.parameters():   # the last residual block
    param.requires_grad = True

optimizer = torch.optim.Adam(trainable_params, lr=0.0001)   # smaller LR
```

Once the new head is reasonably trained, unfreezing more of the network
(starting with the *last* blocks, closest to the output) lets the model
adapt its highest-level features to your specific task. Use a **smaller**
learning rate once more is unfrozen — those layers already have good,
pretrained weights, and you want to nudge them gently rather than
overwrite them with large updates.

**This two-stage recipe is a simplified version of the exact pattern**
used throughout this account's real projects: head-only → last few
blocks → the entire model, with the learning rate shrinking further at
each stage. Same idea, just one more step.

## Run it

```bash
python3 make_shapes.py   # once (same generator as Day 9/10)
python3 main.py
```

Note: images get resized to 224×224 (what resnet18 expects), so this
runs a bit slower than earlier days' 48×48 examples — still fine on CPU
for this small a dataset.

## Exercises

Open `exercises.py` and work through the four TODOs.
