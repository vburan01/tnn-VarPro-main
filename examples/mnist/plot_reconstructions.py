"""
Reconstruction comparison figure.

Usage:
    python plot_reconstructions.py

Before running, make sure RUNS points to the trained standard and VarPro
experiment folders. The script loads the best-validation checkpoints and writes
reconstructions_n*.png into this directory.

Layout (one figure per n_train):
  Row 0 : Original MNIST digits
  Row 1 : Standard reconstruction  +  test loss in row title
  Row 2 : VarPro FC reconstruction +  test loss in row title

8 test images are shown side-by-side.
The best-val-loss checkpoint is loaded for each method.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

# ---------------------------------------------------------------------------
# Make sure the package root is on the path when run as a script
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tnn.layers import Permute, View, tLinearLayer
from tnn.networks import tFullyConnected
from tnn.tensor_utils import dct_matrix
from tnn.training.autoencoder_trainers import AutoencoderWrapper

# ---------------------------------------------------------------------------
# Experiment registry  (same as plot_loss_curves.py)
# ---------------------------------------------------------------------------
N_TRAINS = [1_000, 5_000, 10_000, 20_000, 50_000]

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

BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')

# ---------------------------------------------------------------------------
# Fixed architecture (same for all runs)
# ---------------------------------------------------------------------------
WIDTH      = 10
AUTO_WIDTH = 20
DIM3       = 28
BIAS       = True
N_IMGS     = 8   # number of test images to show

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
device = torch.device('cpu')

def make_dct():
    return dct_matrix(DIM3, dtype=torch.float32, device=device)

def build_standard_net(M):
    """Recreate the baseline autoencoder architecture before loading a checkpoint."""
    net = torch.nn.Sequential(
        View((-1, 28, 28)),
        Permute((1, 0, 2)),
        tFullyConnected((28, AUTO_WIDTH, WIDTH, AUTO_WIDTH, 28), DIM3,
                        M=M, activation=torch.nn.Tanh(),
                        last_activation=None, bias=BIAS),
        Permute((1, 0, 2)),
        View((-1, 1, 28, 28))
    ).to(device)
    return net

def build_varpro_net(M):
    """Recreate the split encoder/decoder VarPro architecture before loading checkpoints."""
    encoder = torch.nn.Sequential(
        View((-1, 28, 28)),
        Permute((1, 0, 2)),
        tFullyConnected((28, AUTO_WIDTH, WIDTH), DIM3,
                        M=M, activation=torch.nn.Tanh(), bias=BIAS)
    ).to(device)
    _dec_fc = tFullyConnected((WIDTH, AUTO_WIDTH, 28), DIM3,
                              M=M, activation=torch.nn.Tanh(),
                              last_activation=None, bias=True).to(device)
    decoder = torch.nn.Sequential(
        _dec_fc,
        Permute((1, 0, 2)),
        View((-1, 1, 28, 28))
    ).to(device)
    return AutoencoderWrapper(encoder, decoder).to(device)

def load_standard(exp_dir, folder):
    """Load the standard Adam checkpoint selected by validation loss."""
    ckpt = os.path.join(BASE, exp_dir, folder, 'best_val_loss_net.pt')
    M = make_dct()
    net = build_standard_net(M)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net

def load_varpro(exp_dir, folder):
    """Load the separate encoder and decoder checkpoints from a VarPro run."""
    enc_ckpt = os.path.join(BASE, exp_dir, folder, 'best_val_loss_encoder.pt')
    dec_ckpt = os.path.join(BASE, exp_dir, folder, 'best_val_loss_decoder.pt')
    M = make_dct()
    net = build_varpro_net(M)
    net.encoder.load_state_dict(torch.load(enc_ckpt, map_location=device))
    net.decoder.load_state_dict(torch.load(dec_ckpt, map_location=device))
    net.eval()
    return net

def get_best_test_loss(csv_path, has_index):
    """Return the test loss at the epoch with the lowest val loss."""
    offset = 1 if has_index else 0
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    # skip header and epoch=-1 row
    data = [[float(v) for v in r] for r in rows[1:] if float(r[offset + 0]) >= 0]
    val_col  = offset + 20
    test_col = offset + 22
    best_row = min(data, key=lambda r: r[val_col])
    return best_row[test_col]

def csv_has_index(csv_path):
    with open(csv_path) as f:
        return f.readline().split(',')[0].strip() == ''

# ---------------------------------------------------------------------------
# Load test data directly from IDX binary (avoid torchvision/NumPy 2.x issue)
# ---------------------------------------------------------------------------
def read_idx_images(path):
    """Parse an IDX3 image file and return a float32 tensor [N, 1, 28, 28] in [0,1]."""
    import struct, gzip
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        data = f.read()
    magic, n, rows, cols = struct.unpack_from('>IIII', data, 0)
    assert magic == 2051, f'Expected IDX3 magic 2051, got {magic}'
    raw = bytearray(data[16:])           # writable copy avoids PyTorch warning
    pixels = torch.frombuffer(raw, dtype=torch.uint8).to(torch.float32)
    return pixels.reshape(n, 1, rows, cols) / 255.0

# Normalisation used during training
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

RAW = os.path.join(DATA_DIR, 'MNIST', 'raw')
test_images_raw = read_idx_images(os.path.join(RAW, 't10k-images-idx3-ubyte'))

# Use the first N_IMGS images (fixed, reproducible)
test_images_raw = test_images_raw[:N_IMGS].to(device)        # [N_IMGS, 1, 28, 28]  raw [0,1]
test_images     = (test_images_raw - MNIST_MEAN) / MNIST_STD  # normalised for model input

# ---------------------------------------------------------------------------
# Build one figure per n_train
# ---------------------------------------------------------------------------
for n in N_TRAINS:
    # Build one reconstruction grid for this training set size.
    std_key    = ('standard', n)
    varpro_key = ('varpro', n)

    # --- load models and get losses ---
    std_exp, std_folder, std_csv   = RUNS[std_key]
    vpr_exp, vpr_folder, vpr_csv   = RUNS[varpro_key]

    std_csv_path = os.path.join(BASE, std_exp, std_folder, std_csv)
    vpr_csv_path = os.path.join(BASE, vpr_exp, vpr_folder, vpr_csv)

    std_net  = load_standard(std_exp, std_folder)
    vpr_net  = load_varpro(vpr_exp, vpr_folder)

    std_test_loss = get_best_test_loss(std_csv_path, csv_has_index(std_csv_path))
    vpr_test_loss = get_best_test_loss(vpr_csv_path, csv_has_index(vpr_csv_path))

    # --- run inference ---
    with torch.no_grad():
        std_recon = std_net(test_images).cpu()   # tensor [N, 1, 28, 28]
        vpr_recon = vpr_net(test_images).cpu()

    # Denormalise back to [0,1] for display
    std_recon_disp = (std_recon * MNIST_STD + MNIST_MEAN).clamp(0, 1)
    vpr_recon_disp = (vpr_recon * MNIST_STD + MNIST_MEAN).clamp(0, 1)
    orig_disp = test_images_raw.cpu()

    # --- plot ---
    fig = plt.figure(figsize=(N_IMGS * 1.5, 5.5))
    fig.suptitle(f'$n_{{train}} = {n:,}$', fontsize=14, y=1.01)

    gs = gridspec.GridSpec(3, N_IMGS, hspace=0.08, wspace=0.04)

    row_labels = [
        'Original',
        f'Standard\n(test loss = {std_test_loss:.4f})',
        f'VarPro FC\n(test loss = {vpr_test_loss:.4f})',
    ]
    rows_data = [orig_disp, std_recon_disp, vpr_recon_disp]

    vmin = 0.0
    vmax = 1.0

    for row_idx, (label, imgs) in enumerate(zip(row_labels, rows_data)):
        for col_idx in range(N_IMGS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(imgs[col_idx].squeeze().tolist(), cmap='gray',
                      vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0,
                              labelpad=72, va='center', ha='right')

    out = os.path.join(OUT_DIR, f'reconstructions_n{n}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved to {out}')

print('Done.')
