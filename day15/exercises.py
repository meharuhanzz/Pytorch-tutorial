"""PyTorch Day 15 -- Capstone Exercises. Run make_shapes.py first if you
haven't. These extend main.py's project rather than starting fresh."""

# 1. Change DROPOUT_P in main.py from 0.3 to 0.6 and re-run. Does the
#    final validation accuracy change? Does the train-val gap (Day 10)
#    look different partway through training?
# TODO (edit main.py's config and re-run, then note your observation
#       here as a comment)

# 2. Change FINETUNE_START_EPOCH from 5 to 2 (unfreeze layer4 much
#    earlier). Does the model reach a good validation loss faster,
#    slower, or does it become unstable?
# TODO

# 3. Add a THIRD stage to main.py's training loop: after layer4 has been
#    fine-tuning for a few epochs, unfreeze layer3 as well (another
#    resnet18 block) at an even smaller learning rate (e.g. 0.00001).
#    This mirrors the full 3-stage recipe used in this account's real
#    projects (head -> last few blocks -> more of the model).
# TODO

# 4. Write a small standalone script (a new file, predict.py) that:
#    loads best_model.pt, loads ONE specific image file from the shapes/
#    folder using PIL + the same transform pipeline as main.py, and
#    prints the predicted class and confidence -- without needing to
#    re-run any training. This is the "use a trained model for a single
#    prediction" pattern you'd use in a real application.
# TODO -- write this as its own predict.py file, not inline here
