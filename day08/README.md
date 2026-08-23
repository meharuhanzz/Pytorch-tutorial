# PyTorch Day 8 — Convolutional Neural Networks

## Why not just flatten the image and use Linear layers?

You could — but you'd throw away something important: **which pixels are
near each other**. A `Linear` layer treats every pixel as equally
(un)related to every other pixel, with no notion that nearby pixels
often form an edge, a texture, or a shape. Convolutional layers are built
specifically to exploit that spatial structure.

## nn.Conv2d — a small filter that slides across the image

```python
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
```

- **`in_channels`** — how many channels the input has (3 for RGB).
- **`out_channels`** — how many different filters to learn. Each one
  produces its own output "feature map" (e.g. one might learn to detect
  vertical edges, another horizontal edges, another a particular color
  transition).
- **`kernel_size`** — the filter's size (3 means a 3x3 window).
- **`padding`** — pads the image edges so the output keeps the same
  height/width as the input (without padding, convolution slightly
  shrinks the output).

The key idea: the **same** learned filter is applied at every position in
the image — this is what lets a Conv2d layer detect "an edge" wherever it
appears, rather than only in one specific spot (which is what would
happen with a Linear layer).

## stride — downsampling as you convolve

```python
nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)
```

`stride=2` moves the filter 2 pixels at a time instead of 1, roughly
halving the output's height and width. This is one common way to shrink
the feature map as the network goes deeper.

## nn.MaxPool2d — another way to downsample

```python
nn.MaxPool2d(kernel_size=2)   # keeps only the max value in each 2x2 window
```

Pooling has no learnable parameters — it's a fixed operation that shrinks
the feature map and adds a bit of robustness to small shifts in the
image (an edge detected slightly off-position still gets picked up).

## The classic CNN block

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))
```

Stack several of these, and a common pattern emerges: each block *shrinks*
the height/width (via pooling) while *growing* the channel count (more
filters) — the network trades spatial resolution for a richer set of
learned features as it goes deeper.

## Flatten — the bridge to Linear layers

```python
nn.Flatten()   # (batch, channels, H, W) -> (batch, channels*H*W)
```

Conv/pool layers work on 4D tensors; `Linear` layers expect 2D
`(batch, features)` input. `Flatten` is the standard way to convert
between the two — always the last step of a CNN's "feature extractor"
before its final classification layers.

## Run it

```bash
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
