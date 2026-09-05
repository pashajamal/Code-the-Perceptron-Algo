# Code the Perceptron Algorithm

An interactive, from-scratch implementation of the **Perceptron algorithm** in Python, with step-by-step visualization of how the decision boundary shifts as the model learns from mislabeled points.

Built with NumPy and Matplotlib — no machine learning libraries required.

## What's Inside

This notebook walks through the Perceptron algorithm from the ground up:

1. **Toy dataset** — a small, linearly separable set of 2D points split into a positive class and a negative class.
2. **Visualization** — an initial scatter plot of the two classes.
3. **Perceptron implementation** — a simple weight-update rule that adjusts the decision boundary whenever it misclassifies a point.
4. **Step-by-step decision boundary plots** — after each iteration, the current decision boundary is plotted so you can watch the line rotate and shift as the algorithm converges.

## How the Algorithm Works

The perceptron learns a weight vector `theta` and bias `theta0` that separate the two classes. For each training step:

- It scans the data points in order.
- If a point is misclassified — i.e. `y[i] * (theta · x[i] + theta0) <= 0` — it updates:
  - `theta += y[i] * x[i]`
  - `theta0 += y[i]`
- It then moves to the next iteration and re-plots the updated decision boundary.

This mirrors the classic online perceptron update rule, applied one misclassified point at a time.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

Open and run the notebook cell by cell:

```bash
jupyter notebook code-the-perceptron-algo.ipynb
```

Each run of the `perceptron()` function will:
- Train on the sample dataset for a set number of steps (default: 10).
- Generate a plot after every iteration showing the current decision boundary against the positive and negative points.

You can experiment by:
- Changing `positive_points` and `negative_points` to your own 2D data.
- Adjusting the `steps` parameter to see how many iterations it takes to converge.

## Example Output

Each iteration produces a plot like:

- Blue points → positive class
- Red points → negative class
- Line → current decision boundary, labeled by iteration number

Watching the sequence of plots shows the boundary rotating toward a valid separation between the two classes.

## License

Feel free to use and adapt this code for learning purposes.
