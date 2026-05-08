"""
Plot validation and test loss vs. epoch for Standard vs. VarPro experiments,
across n_train = 1000, 5000, 10000, 20000, 50000.  (lr = 1e-3 for all)

Usage:
    python plot_loss_curves.py

Before running, make sure RUNS points to the experiment folders you want to plot.
The script writes standard_loss.png, varpro_loss.png, varpro_vs_standard_loss.png,
and loss_n*.png into this directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Experiment registry  {(method, n_train): (subfolder, timestamp, csv_name)}
# ---------------------------------------------------------------------------
RUNS = {
    ('standard', 1_000):  ('tensor_standard',  'MNIST_Standard_1000',  'mnist_tensor_standard.csv'),
    ('standard', 5_000):  ('tensor_standard',  'MNIST_Standard_5000',  'mnist_tensor_standard.csv'),
    ('standard', 10_000): ('tensor_standard',  'MNIST_Standard_10000', 'mnist_tensor_standard.csv'),
    ('standard', 20_000): ('tensor_standard',  'MNIST_Standard_20000', 'mnist_tensor_standard.csv'),
    ('standard', 50_000): ('tensor_standard',  'MNIST_Standard_50000', 'mnist_tensor_standard.csv'),
    ('varpro',   1_000):  ('tensor_varpro_fc', 'MNIST_VarPro_1000',   'mnist_tensor_varpro_fc.csv'),
    ('varpro',   5_000):  ('tensor_varpro_fc', 'MNIST_VarPro_5000',   'mnist_tensor_varpro_fc.csv'),
    ('varpro',   10_000): ('tensor_varpro_fc', 'MNIST_VarPro_10000',  'mnist_tensor_varpro_fc.csv'),
    ('varpro',   20_000): ('tensor_varpro_fc', 'MNIST_VarPro_20000',  'mnist_tensor_varpro_fc.csv'),
    ('varpro',   50_000): ('tensor_varpro_fc', 'MNIST_VarPro_50000',  'mnist_tensor_varpro_fc.csv'),
}

# All paths below are resolved relative to examples/mnist.
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')

N_TRAINS = [1_000, 5_000, 10_000, 20_000, 50_000]

# One colour per n_train, two line styles for the two methods
COLORS  = ['#1f77b4', '#e377c2', '#2ca02c', '#000000', '#7f7f7f']   # blue, pink, green, black, grey
MARKERS = ['o', 's', 's', '*', 's']


def load_losses(csv_path: str):
    """
    Return (epochs, val_loss, test_loss) arrays from a training CSV.

    Column layout (both standard and varpro):
        epoch | lr | betas | betas | eps | weight_decay | amsgrad | maximize |
        foreach | capturable | differentiable | fused | decoupled_weight_decay |
        initial_lr | |params| | |grad| | time | obj |
        train_loss(18) | train_acc(19) | val_loss(20) | val_acc(21) | test_loss(22) | test_acc(23)

    Standard CSVs have an extra unnamed pandas-index column at position 0.
    """
    with open(csv_path) as fh:
        first_col_header = fh.readline().split(',')[0].strip()

    has_index = (first_col_header == '')
    offset = 1 if has_index else 0

    df = pd.read_csv(csv_path, header=0)
    df.columns = range(len(df.columns))

    epoch_col     = offset + 0
    val_loss_col  = offset + 20
    test_loss_col = offset + 22

    df = df[df[epoch_col] >= 0].reset_index(drop=True)
    return (df[epoch_col].values,
            df[val_loss_col].values.astype(float),
            df[test_loss_col].values.astype(float))


# keep old name as alias for the combined overview
def load_val_loss(csv_path):
    epochs, val, _ = load_losses(csv_path)
    return epochs, val


# ---------------------------------------------------------------------------
# Per-sample-size comparison plots  —  val loss (left) | test loss (right)
# ---------------------------------------------------------------------------
METHOD_STYLES = {
    'standard': dict(color='#1f77b4', linestyle='-',  marker='o', label='Standard',
                     markerfacecolor='none'),
    'varpro':   dict(color='#d62728', linestyle='--', marker='s', label='VarPro FC',
                     markerfacecolor='#d62728'),
}

out_dir = os.path.dirname(os.path.abspath(__file__))

for n in N_TRAINS:
    # Build one standard-vs-VarPro test-loss plot for this training set size.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_title(f'$n_{{train}} = {n:,}$', fontsize=13)

    for method, sty in METHOD_STYLES.items():
        key = (method, n)
        if key not in RUNS:
            continue
        exp_dir, folder, csv_file = RUNS[key]
        csv_path = os.path.join(BASE, exp_dir, folder, csv_file)
        if not os.path.exists(csv_path):
            print(f'[WARNING] missing CSV: {csv_path}')
            continue

        epochs, _, test_loss = load_losses(csv_path)
        best_loss  = float(test_loss.min())
        best_epoch = int(epochs[test_loss.argmin()])
        markevery  = max(1, len(epochs) // 10)

        ax.semilogy(
            epochs, test_loss,
            color=sty['color'], linestyle=sty['linestyle'],
            marker=sty['marker'], markevery=markevery, markersize=7,
            markerfacecolor=sty['markerfacecolor'],
            markeredgecolor=sty['color'],
            linewidth=1.8,
            label=f"{sty['label']}  (best={best_loss:.3f} @ ep {best_epoch})",
        )
        ax.plot(best_epoch, best_loss,
                marker='*', markersize=14,
                color=sty['color'], markeredgecolor='k', markeredgewidth=0.5,
                zorder=5, linestyle='none')

    ax.set_xlabel('epoch', fontsize=12)
    ax.set_ylabel('test loss', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.45)
    fig.tight_layout()
    out = os.path.join(out_dir, f'loss_n{n}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved to', out)

# ---------------------------------------------------------------------------
# Combined overview (all n_train, Standard solid vs VarPro dashed)
# ---------------------------------------------------------------------------
COLORS  = ['#1f77b4', '#e377c2', '#2ca02c', '#000000', '#7f7f7f']
MARKERS = ['o', 's', 's', '*', 's']

fig0, ax0 = plt.subplots(figsize=(7, 4.5))
for i, n in enumerate(N_TRAINS):
    # Overlay every training set size in one compact comparison figure.
    color, marker = COLORS[i], MARKERS[i]
    for method, linestyle, mfc in [('standard', '-', 'none'), ('varpro', '--', color)]:
        key = (method, n)
        if key not in RUNS:
            continue
        exp_dir, folder, csv_file = RUNS[key]
        csv_path = os.path.join(BASE, exp_dir, folder, csv_file)
        if not os.path.exists(csv_path):
            continue
        epochs, _, test_loss = load_losses(csv_path)
        markevery = max(1, len(epochs) // 10)
        label = f'$n={n:,}$ {"VarPro" if method=="varpro" else "Std"}'
        ax0.semilogy(epochs, test_loss, color=color, linestyle=linestyle,
                     marker=marker, markevery=markevery, markersize=7,
                     markerfacecolor=mfc, markeredgecolor=color,
                     label=label, linewidth=1.5)
ax0.set_xlabel('epoch', fontsize=12)
ax0.set_ylabel('test loss', fontsize=12)
ax0.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax0.legend(fontsize=7.5, ncol=2, loc='upper right', framealpha=0.9)
ax0.grid(True, which='both', linestyle=':', alpha=0.45)
fig0.tight_layout()
out0 = os.path.join(out_dir, 'varpro_vs_standard_loss.png')
fig0.savefig(out0, dpi=150, bbox_inches='tight')
print('Saved to', out0)

plt.show()
