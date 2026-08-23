# PyTorch Day 7 — Real Image Data

Days 1-6 used tensors and synthetic 2D points. Today: real image files
on disk, the way every image project in this repo actually works.

**Run `python3 make_sample_images.py` once first** — it generates a
tiny 3-class dataset of small synthetic images under `sample_images/`,
so this lesson doesn't depend on downloading anything.

## transforms — PIL image to model-ready tensor

An image loaded from disk is a `PIL.Image`, not a tensor. `transforms`
convert and preprocess it into what a model needs:

```python
transforms.ToTensor()   # PIL Image -> tensor, rearranges to (C, H, W),
                          # scales pixel values from [0, 255] to [0.0, 1.0]
```

## Compose — chaining a full pipeline

```python
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])
```

That specific mean/std is the ImageNet dataset's statistics — used
throughout this repo's real projects because it matches what pretrained
models (Day 11) were themselves trained with. After `Normalize`, pixel
values are no longer in `[0, 1]` — that's expected, not a bug.

## Data augmentation — only for training data

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.3, contrast=0.3)
```

Small random distortions applied during training teach the model to be
robust to variation it'll see in the real world. Run the same augment
pipeline on the same image twice and you'll get *different* tensors each
time — that randomness is the point. **Never apply augmentation to
validation/test data** — you want consistent, repeatable numbers there.

## ImageFolder — loading a labeled dataset from a folder structure

The convention: one subfolder per class.

```
sample_images/
  red_ish/
    red_ish_0.png
    red_ish_1.png
    ...
  green_ish/
    ...
  blue_ish/
    ...
```

```python
dataset = ImageFolder("sample_images", transform=preprocess)
```

`ImageFolder` reads that structure automatically — no manual labeling
needed, folder names become the classes. This is exactly the format
used by this repo's medicinal-plant image classifier.

## Separate transforms for train vs. val

```python
train_dataset = ImageFolder(DATA_DIR, transform=train_tfms)  # includes augmentation
val_dataset = ImageFolder(DATA_DIR, transform=val_tfms)        # resize/normalize only
```

Same files, two different pipelines — this exact pattern (`train_tfms` /
`val_tfms`) is what the plant classifier project in this repo uses.

## Run it

```bash
python3 make_sample_images.py   # once
python3 main.py
```

## Exercises

Open `exercises.py` and work through the four TODOs.
