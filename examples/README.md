## Examples

The main project files for the MNIST VarPro autoencoder report are in:

```text
mnist/
```

Key MNIST files:

- `mnist/ex_mnist_tensor_standard_regression.py`: baseline t-NN autoencoder trained with Adam.
- `mnist/ex_mnist_tensor_varpro_regression.py`: t-NN autoencoder trained with VarPro on the final decoder layer.
- `mnist/plot_loss_curves.py`: loss figures used in the report.
- `mnist/plot_reconstructions.py`: reconstruction figures used in the appendix.

## Running MNIST Report Experiments

From the repository root, a standard baseline run looks like:

```console
python -m examples.mnist.ex_mnist_tensor_standard_regression --max_epochs 200 --n_train 1000 --n_val 10000 --n_test 10000 --width 10 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9 --step_size 100 --M dct --seed 42
```

A VarPro run looks like:

```console
python -m examples.mnist.ex_mnist_tensor_varpro_regression --max_epochs 200 --n_warmup 10 --n_train 1000 --n_val 10000 --n_test 10000 --width 10 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9 --step_size 100 --M dct --seed 42
```

Change `--n_train` to `5000`, `10000`, `20000`, or `50000` to reproduce the other
training-size experiments.
