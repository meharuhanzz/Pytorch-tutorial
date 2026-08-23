"""PyTorch Day 12 -- Saving, Loading & Checkpointing.

Run me with:  python3 main.py
"""
import os

import torch
from torch import nn

torch.manual_seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))


# A small model to demonstrate with.
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


# ---- 1. state_dict -- a model's weights, as a plain dictionary ----
# Every nn.Module has a .state_dict(): an ordered dict mapping each
# parameter's name to its tensor value. This is the RECOMMENDED thing to
# save -- not the whole model object (see section 5 for why).
print("=== state_dict ===")
model = SmallNet()
state = model.state_dict()
for name, tensor in state.items():
    print(f"  {name}: shape {tuple(tensor.shape)}")

# ---- 2. Saving and loading weights ----
print("\n=== Saving & loading weights ===")
weights_path = os.path.join(HERE, "small_net_weights.pt")
torch.save(model.state_dict(), weights_path)
print(f"Saved to {weights_path}")

# To load: create a model with the SAME architecture, then load the
# saved weights into it. The architecture itself isn't saved -- your
# code defining SmallNet is what recreates the structure.
new_model = SmallNet()
new_model.load_state_dict(torch.load(weights_path, weights_only=True))
print("Loaded weights into a fresh SmallNet instance")

# Confirm they now produce identical output on the same input:
sample_input = torch.rand(1, 4)
with torch.no_grad():
    out1 = model(sample_input)
    out2 = new_model(sample_input)
print(f"Original model output:  {out1}")
print(f"Loaded model output:    {out2}")
print(f"Identical: {torch.equal(out1, out2)}")

# ---- 3. Saving a full checkpoint -- for resuming training ----
# If you want to PAUSE and RESUME training later (not just use the model
# for predictions), you need more than the weights: the optimizer's
# state (e.g. Adam's per-parameter moving averages) and which epoch you
# stopped at. A checkpoint is usually a dict bundling all of this.
print("\n=== A full training checkpoint ===")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = {
    "epoch": 5,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": 0.234,
}
checkpoint_path = os.path.join(HERE, "checkpoint.pt")
torch.save(checkpoint, checkpoint_path)
print(f"Saved checkpoint (epoch, model weights, optimizer state, best_val_loss) to {checkpoint_path}")

# ---- 4. Resuming from a checkpoint ----
print("\n=== Resuming from a checkpoint ===")
resumed_model = SmallNet()
resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)

loaded = torch.load(checkpoint_path, weights_only=True)
resumed_model.load_state_dict(loaded["model_state_dict"])
resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])
start_epoch = loaded["epoch"]
best_val_loss = loaded["best_val_loss"]

print(f"Resuming from epoch {start_epoch}, best_val_loss so far: {best_val_loss}")
print("You'd now continue a training loop with: for epoch in range(start_epoch, num_epochs): ...")

# ---- 5. Why save state_dict, not the whole model object ----
# torch.save(model, path) IS possible -- it pickles the entire Python
# object, architecture and all. This seems convenient (no need to
# recreate the class before loading) but is fragile: if your model's
# class definition changes at all later (even a minor library version
# bump changing an internal attribute), loading an old pickled model can
# break in confusing ways. Saving state_dict -- just the numbers -- and
# recreating the architecture from your current code is the robust,
# recommended approach used throughout this account's real projects.
print("\n=== state_dict vs whole-model saving ===")
print("Recommended:   torch.save(model.state_dict(), path)")
print("               model = SmallNet(); model.load_state_dict(torch.load(path))")
print("Fragile:       torch.save(model, path)  # pickles the whole object")
print("               model = torch.load(path)   # can break across library versions")

# ---- 6. The "save only the best model" pattern ----
# This is exactly the pattern used in this account's real training
# scripts: track the best validation loss seen so far, and only
# overwrite your saved checkpoint when the model actually improves --
# so you always end up with the best-performing version, not just
# whatever the last epoch happened to produce.
print("\n=== Save-best-only pattern (pseudocode) ===")
print("""
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')   # overwrite only on improvement
""")

# ---- cleanup ----
os.remove(weights_path)
os.remove(checkpoint_path)
print("Cleaned up the demo files created above.")
