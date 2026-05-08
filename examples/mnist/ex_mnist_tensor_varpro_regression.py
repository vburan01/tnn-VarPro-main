"""
Train the MNIST t-NN autoencoder with VarPro on the final decoder layer.

Typical run for reproducibility:
 python -m examples.mnist.ex_mnist_tensor_varpro_regression
 --max_epochs 200 --n_warmup 10 --n_train 1000 --n_val 10000 --n_test 10000
 --width 10 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9
 --step_size 100 --max_grad_norm 1.0 --M dct --seed 42

Change --n_train to 5000, 10000, 20000, or 50000 to reproduce the other
training-size experiments. Results are written under experiments/tensor_varpro_fc/.
"""

import torch
from tnn.layers import Permute, View, tLinearLayer
from tnn.networks import tFullyConnected, tHamiltonianResNet
from tnn.training.autoencoder_trainers import train_tnn_varpro_autoencoder, AutoencoderWrapper
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
import os
import datetime
import pickle
import pandas as pd
from examples.autoencoder.setup_mnist import setup_mnist

# Read command-line experiment settings and add the optional Hamiltonian encoder flag.
parser = setup_parser()
parser.add_argument('--hamiltonian', action='store_true', help='use Hamiltonian encoder instead of fully-connected')
args = parser.parse_args()

# Fix all random seeds so repeated runs are comparable.
seed_everything(args.seed)

# Load MNIST and then wrap labels away so the target is the image itself.
train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size, args.data_dir)

# --- Wrap DataLoaders for regression (autoencoder) ---
from torch.utils.data import DataLoader
class RegressionWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, x

train_loader = DataLoader(RegressionWrapper(train_loader.dataset), batch_size=args.batch_size)
val_loader = DataLoader(RegressionWrapper(val_loader.dataset), batch_size=args.batch_size)
test_loader = DataLoader(RegressionWrapper(test_loader.dataset), batch_size=args.batch_size)

# Use CUDA when available, otherwise run on CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build the mode-3 transform used by every t-NN layer.
dim3 = 28
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
elif args.M == 'data':
    A = torch.empty(0)
    for data in train_loader:
        A = torch.cat((A, data[0]), dim=0)
    A = A.reshape(-1, A.shape[-1]).T
    M, _, _ = torch.linalg.svd(A, full_matrices=False)
    M = M.T.to(dtype=torch.float32, device=device)
    del A
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)

# Build encoder and decoder separately so the final decoder layer can be solved
# by VarPro while the earlier layers remain Adam-trained.
# Matches paper t-NN(auto_width, width): 2-layer encoder 28->auto_width->width,
# 2-layer decoder width->auto_width->28 (last layer solved by VarPro, no activation).
#
# Encoder options:
#   FC:          View -> Permute -> tFullyConnected([28, auto_width, width], Tanh)
#   Hamiltonian: View -> Permute -> tLinearLayer(28 -> width, Tanh) -> tHamiltonianResNet(width, depth)
#                (Hamiltonian block keeps width fixed; auto_width not used in this path)
if args.hamiltonian:
    encoder = torch.nn.Sequential(
        View((-1, 28, 28)),
        Permute((1, 0, 2)),
        tLinearLayer(28, args.width, dim3, M=M, activation=torch.nn.Tanh(), bias=args.bias),
        tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                           depth=args.depth, h=args.h_step, activation=torch.nn.Tanh())
    ).to(device)
else:
    encoder = torch.nn.Sequential(
        View((-1, 28, 28)),
        Permute((1, 0, 2)),
        tFullyConnected((28, args.auto_width, args.width), dim3, M=M, activation=torch.nn.Tanh(), bias=args.bias)
    ).to(device)
# Decoder: width -> auto_width (Tanh, updated by Adam) -> 28 (linear, VarPro solves this)
# last_activation=None ensures only the final layer has no activation.
_dec_fc = tFullyConnected((args.width, args.auto_width, 28), dim3, M=M,
                          activation=torch.nn.Tanh(), last_activation=None, bias=True).to(device)
decoder = torch.nn.Sequential(
    _dec_fc,
    Permute((1, 0, 2)),
    View((-1, 1, 28, 28))
).to(device)

net = AutoencoderWrapper(encoder, decoder).to(device)

# MSE reconstruction loss. During the VarPro phase, Adam updates the encoder and
# the intermediate decoder layer only; the final decoder layer is least-squares solved.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    list(net.encoder.parameters()) + list(_dec_fc.layers[:-1].parameters()),
    lr=args.lr, weight_decay=args.weight_decay,
    betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Create a timestamped experiment folder with logs and checkpoints.
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
arch = 'hamiltonian' if args.hamiltonian else 'fc'
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', f'tensor_varpro_{arch}', start_time)
filename = f'mnist_tensor_varpro_{arch}.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'mnist_tensor_varpro')
logger.info(f'args: {args}')
logger.info("---------------------- Network ----------------------------")
logger.info(net)
logger.info("Number of trainable parameters: {}".format(number_network_weights(net)))
logger.info("--------------------------------------------------")
logger.info(str(optimizer))
logger.info(str(scheduler))
logger.info("dtype={:} device={:}".format(train_loader.dataset.dataset.data.dtype, device))
logger.info("epochs={:} ".format(args.max_epochs))
logger.info("saveLocation = {:}".format(sPath))
logger.info("--------------------------------------------------\n")

# Run the warm-up and alternating VarPro training loop.
import time as _time
if torch.cuda.is_available():
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    results = train_tnn_varpro_autoencoder(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, n_warmup=args.n_warmup, alpha2=args.weight_decay, max_grad_norm=args.max_grad_norm, scheduler=scheduler, device=device, logger=logger, sPath=sPath, M=M
    )
    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) / 1000.0
else:
    t0 = _time.perf_counter()
    results = train_tnn_varpro_autoencoder(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, n_warmup=args.n_warmup, alpha2=args.weight_decay, max_grad_norm=args.max_grad_norm, scheduler=scheduler, device=device, logger=logger, sPath=sPath, M=M
    )
    t1 = _time.perf_counter()
    elapsed = t1 - t0
logger.info('Total Training Time: {:.2f} seconds'.format(elapsed))
torch.save(net.state_dict(), sPath + '/last_net.pt')

results = results[-1] if isinstance(results, tuple) else results

# ---- Load best checkpoint (by val loss) before final evaluation ----
net.encoder.load_state_dict(torch.load(os.path.join(sPath, 'best_val_loss_encoder.pt'), map_location=device))
net.decoder.load_state_dict(torch.load(os.path.join(sPath, 'best_val_loss_decoder.pt'), map_location=device))
logger.info(f'Loaded best checkpoint (epoch {results["best_val_loss_epoch"]}, val_loss={results["best_val_loss"]:.4f}) for final test evaluation.')

# ---- Compute final test loss ----
criterion.reduction = 'mean'
net.eval()
test_loss = 0.0
n_test = 0
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        test_loss += criterion(net(xb), xb).item() * xb.size(0)
        n_test += xb.size(0)
test_loss /= n_test

# ---- Bundle everything into a single results dict ----
results['args']             = args
results['arch']             = arch                         # 'fc' or 'hamiltonian'
results['n_params']         = number_network_weights(net)
results['elapsed_sec']      = elapsed
results['test_loss_final']  = test_loss
results['encoder_str']      = str(net.encoder)
results['decoder_str']      = str(net.decoder)

# training history as a tidy DataFrame (epoch, train_loss, val_loss, ...)
his_df = pd.DataFrame(results['val'], columns=results['str'])
if 'epoch' not in his_df.columns:
    his_df.insert(0, 'epoch', range(len(his_df)))
results['history_df'] = his_df

pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
his_df.to_csv(os.path.join(sPath, filename[:-4] + '.csv'), index=False)
logger.info(f'Final test loss: {test_loss:.4f}')
logger.info(f'Results saved to {sPath}')

# Optionally: visualize reconstructions
import matplotlib.pyplot as plt
x, _ = next(iter(test_loader))
x = x.to(device)
z = net(x).cpu().detach()
vmax = x.max()
vmin = x.min()
plt.figure()
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(x[i].squeeze().cpu(), vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 8, i + 9)
    plt.imshow(z[i].squeeze(), vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
plt.savefig(sPath + '/reconstructions.png')
plt.show()
