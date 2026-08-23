"""PyTorch Day 8 -- Convolutional Neural Networks (CNNs).

Run me with:  python3 main.py
"""
import torch
from torch import nn

torch.manual_seed(42)

# ---- 1. Why not just use nn.Linear for images? ----
# A small 64x64 RGB image has 64*64*3 = 12,288 values. Flattening it and
# feeding it to nn.Linear would work, but it throws away something
# important: which pixels are NEAR each other. A Linear layer treats
# pixel (0,0) and pixel (63,63) as no more related than any other pair --
# it has no concept of "nearby pixels tend to be related" (an edge, a
# texture, a shape). Convolutional layers are built specifically to
# exploit that spatial structure.

# ---- 2. nn.Conv2d -- sliding a small filter across an image ----
# A Conv2d layer learns a small filter (e.g. 3x3) that slides across the
# whole image, computing a weighted sum at each position. The SAME filter
# is reused at every position -- this is what lets it detect "an edge"
# wherever it appears in the image, not just in one specific spot.
print("=== nn.Conv2d ===")
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
# in_channels=3: input is RGB (3 color channels)
# out_channels=8: learn 8 different filters -> 8 output "feature maps"
# kernel_size=3: each filter is 3x3
# padding=1: pad the image edges so output stays the same H/W as input

fake_image = torch.rand(1, 3, 64, 64)   # (batch, channels, height, width)
output = conv(fake_image)
print(f"input shape:  {fake_image.shape}")
print(f"output shape: {output.shape}")   # (1, 8, 64, 64) -- 8 feature maps now, same H/W

# ---- 3. What padding and stride change ----
print("\n=== padding & stride ===")
no_padding = nn.Conv2d(3, 8, kernel_size=3, padding=0)
print(f"no padding: {fake_image.shape} -> {no_padding(fake_image).shape}")   # shrinks slightly

strided = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)
print(f"stride=2:   {fake_image.shape} -> {strided(fake_image).shape}")   # halves H and W
# stride=2 means the filter jumps 2 pixels at a time instead of 1 --
# a common way to downsample (shrink) the feature map as you go deeper.

# ---- 4. Pooling -- another way to downsample ----
# MaxPool2d shrinks the feature map by keeping only the largest value in
# each small window -- reduces size and adds a little robustness to
# small shifts in the image, with no learnable parameters at all.
print("\n=== nn.MaxPool2d ===")
pool = nn.MaxPool2d(kernel_size=2)   # 2x2 windows, keep the max of each
pooled = pool(output)
print(f"before pooling: {output.shape}")
print(f"after 2x2 max pool: {pooled.shape}")   # height and width both halved

# ---- 5. Stacking Conv + activation + pool -- the classic CNN block ----
print("\n=== A CNN feature extractor block ===")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


block = ConvBlock(3, 16)
out = block(fake_image)
print(f"ConvBlock(3, 16): {fake_image.shape} -> {out.shape}")

# ---- 6. A complete small CNN classifier ----
# Stack several ConvBlocks (each one shrinks H/W, grows the channel
# count) then Flatten and finish with Linear layers -- this is the
# classic CNN shape.
print("\n=== A complete CNN ===")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16),    # 64x64 -> 32x32
            ConvBlock(16, 32),   # 32x32 -> 16x16
            ConvBlock(32, 64),   # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (batch, 64, 8, 8) -> (batch, 64*8*8)
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes=3)
print(model)

batch = torch.rand(4, 3, 64, 64)   # a batch of 4 images
logits = model(batch)
print(f"\ninput: {batch.shape} -> output logits: {logits.shape}")   # (4, 3) -- 3 class scores per image

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ---- 7. Flatten -- the bridge between conv layers and Linear layers ----
# nn.Flatten() takes a (batch, channels, height, width) tensor and
# collapses everything except the batch dimension into one long vector
# per example -- Linear layers expect 2D input (batch, features), not 4D.
print("\n=== Flatten ===")
example = torch.rand(2, 64, 8, 8)
flattened = nn.Flatten()(example)
print(f"{example.shape} -> Flatten -> {flattened.shape}")
