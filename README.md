# Variable Projection for Tensor Neural Network Autoencoders

This repository extends the matrix-mimetic tensor neural network (`tNN`) codebase
introduced by Newman, Horesh, Avron, and Kilmer in *Stable tensor neural networks
for efficient deep learning*. The original tNN framework replaces dense matrix
layers with transform-based tensor `\star_M` layers. This project builds on that
framework by adding a variable projection (VarPro) training method for a tNN
autoencoder on MNIST.

The main experiment compares:

- a baseline tNN autoencoder trained with Adam on all weights, and
- a VarPro tNN autoencoder that solves the final decoder layer by least squares
  while training the remaining layers with Adam.

## Acknowledgment

This project is based on the original tNN implementation and paper:

```bibtex
@article{newman_stable_2024,
  title   = {Stable tensor neural networks for efficient deep learning},
  author  = {Newman, Elizabeth and Horesh, Lior and Avron, Haim and Kilmer, Misha E.},
  journal = {Frontiers in Big Data},
  volume  = {7},
  pages   = {1363978},
  year    = {2024},
  doi     = {10.3389/fdata.2024.1363978}
}
```

The paper is available here:
https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1363978/full

## Installation

From the repository root:

```console
python -m pip install -e .
```

The experiments require PyTorch, NumPy, pandas, matplotlib, and SciPy. MNIST is
downloaded automatically by the setup script when needed.

## Main Files

```text
examples/mnist/ex_mnist_tensor_standard_regression.py
examples/mnist/ex_mnist_tensor_varpro_regression.py
examples/mnist/plot_loss_curves.py
examples/mnist/plot_reconstructions.py
tnn/training/autoencoder_trainers.py
```

The VarPro training logic is implemented in:

```text
tnn/training/autoencoder_trainers.py
```

## Reproducing the MNIST Experiments

Run commands from the repository root.

### Standard Adam Baseline

For `n_train = 1000`:

```console
python -m examples.mnist.ex_mnist_tensor_standard_regression --max_epochs 200 --n_train 1000 --n_val 10000 --n_test 10000 --width 10 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9 --step_size 100 --M dct --seed 42
```

Repeat with:

```text
--n_train 5000
--n_train 10000
--n_train 20000
--n_train 50000
```

### VarPro Training

For `n_train = 1000`:

```console
python -m examples.mnist.ex_mnist_tensor_varpro_regression --max_epochs 200 --n_warmup 10 --n_train 1000 --n_val 10000 --n_test 10000 --width 10 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9 --step_size 100 --max_grad_norm 1.0 --M dct --seed 42
```

Repeat with:

```text
--n_train 5000
--n_train 10000
--n_train 20000
--n_train 50000
```

## Regenerating Figures

After the experiment folders exist and the `RUNS` dictionaries point to them:

```console
python examples/mnist/plot_loss_curves.py
python examples/mnist/plot_reconstructions.py
```

These scripts generate the loss plots and reconstruction images used in the
report.

## Notes

The older paper-style example files are kept under `examples/legacy_paper_examples/`
for reference. The MNIST VarPro autoencoder workflow is in `examples/mnist/`.
