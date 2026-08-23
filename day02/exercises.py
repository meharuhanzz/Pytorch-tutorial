"""PyTorch Day 2 -- Exercises. Fill in the TODOs, then run: python3 exercises.py"""
import torch

# 1. Create x = torch.tensor(5.0, requires_grad=True). Compute
#    y = 3 * x**2 + 2 * x + 1, call .backward(), and print x.grad.
#    By hand, dy/dx = 6x + 2 -- at x=5 that's 32. Confirm it matches.
# TODO

# 2. Create two tensors a and b (requires_grad=True) with values of your
#    choice. Compute c = a * b + a**2, call c.backward(), and print
#    a.grad and b.grad. Work out the expected values by hand
#    (dc/da = b + 2a, dc/db = a) and confirm they match.
# TODO

# 3. Demonstrate the gradient-accumulation gotcha yourself: create
#    w = torch.tensor(2.0, requires_grad=True), run loss = w**3 through
#    .backward() twice IN A ROW without zeroing, and print w.grad after
#    each call to see it accumulate. Then zero it and confirm it resets.
# TODO

# 4. Run the manual gradient descent loop from main.py, but change the
#    target from 10 to -5 (i.e. minimize (w + 5)**2) and the learning
#    rate to 0.2. Run it for 8 steps and print w at each step -- confirm
#    it approaches -5.
# TODO
