"""PyTorch Day 7 -- Real Image Data: transforms & ImageFolder.

Run make_sample_images.py FIRST if you haven't:
    python3 make_sample_images.py
    python3 main.py
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "sample_images")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("Run 'python3 make_sample_images.py' first to generate sample images.")

# ---- 1. transforms -- turning a PIL image into a model-ready tensor ----
# Real images come in as files (PNG/JPEG), loaded as PIL Image objects --
# not tensors. transforms convert and preprocess them into what a model
# actually needs: a normalized tensor of a fixed size.
print("=== Basic transforms ===")
sample_path = os.path.join(DATA_DIR, "red_ish", "red_ish_0.png")
pil_image = Image.open(sample_path)
print(f"Loaded a PIL image: size={pil_image.size}, mode={pil_image.mode}")

to_tensor = transforms.ToTensor()
tensor_image = to_tensor(pil_image)
print(f"After ToTensor(): shape={tensor_image.shape}, dtype={tensor_image.dtype}")
print(f"pixel value range: [{tensor_image.min():.3f}, {tensor_image.max():.3f}]")
# Note: ToTensor() also rearranges from (H, W, Channels) to (Channels, H, W)
# -- PyTorch's convention -- and scales pixel values from [0, 255] to [0.0, 1.0].

# ---- 2. Chaining transforms with Compose ----
# Compose runs a list of transforms in order -- this is the standard way
# to build a full preprocessing pipeline.
print("\n=== transforms.Compose ===")
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),        # standardize every image to the same size
    transforms.ToTensor(),
    transforms.Normalize(               # standard ImageNet mean/std -- used
        mean=[0.485, 0.456, 0.406],      # throughout this repo's real projects
        std=[0.229, 0.224, 0.225],
    ),
])

processed = preprocess(pil_image)
print(f"After full pipeline: shape={processed.shape}")
print(f"pixel value range: [{processed.min():.3f}, {processed.max():.3f}]")
# Normalize shifts values outside [0, 1] -- this is expected and correct;
# it centers the data in a range that tends to train better, matching
# the statistics the pretrained models you'll use in Day 11 expect.

# ---- 3. Data augmentation -- artificially expanding your training data ----
# Applied only to TRAINING data, never to validation/test data. Small
# random distortions teach the model to be robust to variations it'll
# see in the real world (slightly rotated, flipped, different lighting).
print("\n=== Data augmentation ===")
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Run the SAME image through the augmentation twice -- notice the
# results differ, since the transforms are randomized:
aug1 = augment(pil_image)
aug2 = augment(pil_image)
print(f"Same source image, two augmented versions -- identical tensors? {torch.equal(aug1, aug2)}")

# ---- 4. ImageFolder -- loading a whole labeled dataset from disk ----
# The convention: one subfolder per class, images inside. ImageFolder
# reads this structure automatically and assigns labels based on folder
# names -- this is exactly the format used by this repo's medicinal
# plant classifier project.
print("\n=== ImageFolder ===")
print(f"Directory structure expected:")
print(f"  {DATA_DIR}/")
for class_name in sorted(os.listdir(DATA_DIR)):
    print(f"    {class_name}/")

dataset = ImageFolder(DATA_DIR, transform=preprocess)
print(f"\nFound {len(dataset)} images across {len(dataset.classes)} classes")
print(f"Classes: {dataset.classes}")
print(f"class_to_idx: {dataset.class_to_idx}")

image, label = dataset[0]
print(f"dataset[0]: image shape {image.shape}, label {label} ({dataset.classes[label]})")

# ---- 5. Putting it together with a DataLoader (Day 6) ----
print("\n=== ImageFolder + DataLoader ===")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch_images, batch_labels = next(iter(loader))
print(f"batch shape: {batch_images.shape}")   # (batch_size, channels, height, width)
print(f"batch labels: {batch_labels.tolist()}")

# ---- 6. Separate transforms for train vs. validation ----
# This is the standard real-world pattern: augment during training,
# but only resize/normalize (no randomness) during validation, so your
# validation numbers are consistent and repeatable run to run.
print("\n=== Train vs. val transforms ===")
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
val_tfms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(DATA_DIR, transform=train_tfms)
val_dataset = ImageFolder(DATA_DIR, transform=val_tfms)
print("Same underlying files, two different transform pipelines -- "
      "this is exactly the pattern this repo's plant classifier uses "
      "for its train_tfms / val_tfms.")
