"""PyTorch Day 14 -- Exercises. Run make_shapes.py first if you haven't.
Fill in the TODOs, then run: python3 exercises.py"""
from sklearn.metrics import confusion_matrix, classification_report

# 1. Given these made-up predictions and true labels for a 3-class
#    problem (0, 1, 2), compute and print the confusion matrix:
true_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
predictions = [0, 0, 1, 1, 1, 2, 2, 0, 2]
# TODO

# 2. From the confusion matrix in exercise 1, compute per-class accuracy
#    BY HAND (i.e. write the arithmetic yourself, don't just call a
#    library function) for each of the 3 classes, then confirm your
#    numbers match cm.diagonal() / cm.sum(axis=1).
# TODO

# 3. Print classification_report for the same true_labels/predictions.
#    Which class has the lowest recall? What does that tell you about
#    what the model in this made-up example is getting wrong?
# TODO

# 4. Run main.py's full pipeline (training + evaluation) TWICE with two
#    different values of torch.manual_seed() at the top (e.g. 1 and 2).
#    Do you get the same confusion matrix both times? Why or why not --
#    write your answer as a comment (hint: think about what else besides
#    the seed affects the random train/val split and weight
#    initialization).
# TODO
