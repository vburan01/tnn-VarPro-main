"""
Train the baseline MNIST t-NN autoencoder with standard Adam.

Typical run for reproducibility:
 python -m examples.mnist.ex_mnist_tensor_standard_regression
 --max_epochs 200 --n_train 1000 --n_val 10000 --n_test 10000 --width 10
 --auto_width 20 --batch_size 32 --lr 1e-3 --gamma 0.9 --step_size 100
 --M dct --seed 42

Change --n_train to 5000, 10000, 20000, or 50000 to reproduce the other
training-size experiments. Results are written under experiments/tensor_standard/.
"""

import torch
from tnn.layers import Permute, View, tLinearLayer
from tnn.networks import tFullyConnected
from tnn.networks.t_fully_connected import _UNSET
from tnn.training.batch_train import train
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
import os
import datetime
import pickle
import pandas as pd
from examples.autoencoder.setup_mnist import setup_mnist 

# Read command-line experiment settings from the shared MNIST parser.
parser = setup_parser()
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
print(f"DEBUG: dim3={dim3}, M.shape={M.shape}")

# Build the full autoencoder as one Sequential model.
# Architecture: 28 -> auto_width -> width -> auto_width -> 28.
# In the standard baseline, Adam trains every layer including the final decoder.
net = torch.nn.Sequential(
    View((-1, 28, 28)),
    Permute((1, 0, 2)),
    tFullyConnected((28, args.auto_width, args.width, args.auto_width, 28), dim3, M=M, activation=torch.nn.Tanh(), last_activation=None, bias=args.bias),
    Permute((1, 0, 2)),
    View((-1, 1, 28, 28))
).to(device)

# MSE reconstruction loss with Adam on all network parameters.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Create a timestamped experiment folder with logs and checkpoints.
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'tensor_standard', start_time)
filename = 'mnist_tensor_standard.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'mnist_tensor_standard')
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

import time as _time
if torch.cuda.is_available():
    torch.cuda.synchronize()

# --- DEBUG: Print output and target shapes before loss computation ---
def debug_train(*args, **kwargs):
    # Wrap the original train function to print shapes
    from tnn.training import batch_train
    orig_train_one_epoch = batch_train.train_one_epoch
    def wrapped_train_one_epoch(*a, **kw):
        for batch in a[2]:  # train_loader
            data, target = batch
            output = a[0](data.to(a[6]))  # net, device
            print('DEBUG: output.shape =', output.shape, 'target.shape =', target.shape)
            break
        return orig_train_one_epoch(*a, **kw)
    batch_train.train_one_epoch = wrapped_train_one_epoch

# --- Training logic ---

# Print one input/output shape check before the long training loop.
first_batch = next(iter(train_loader))
if isinstance(first_batch, (list, tuple)):
    x = first_batch[0]
else:
    x = first_batch
print('DEBUG: input batch shape before model:', x.shape)
with torch.no_grad():
    x = x.to(device)
    z = net(x)
    print('DEBUG: output batch shape after model:', z.shape)

if torch.cuda.is_available():
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    results = train(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, scheduler=scheduler, device=device, logger=logger, sPath=sPath
    )
    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) / 1000.0
else:
    t0 = _time.perf_counter()
    results = train(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, scheduler=scheduler, device=device, logger=logger, sPath=sPath
    )
    t1 = _time.perf_counter()
    elapsed = t1 - t0

logger.info('Total Training Time: {:.2f} seconds'.format(elapsed))
torch.save(net.state_dict(), sPath + '/last_net.pt')

results['args'] = args
pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(pd.DataFrame(results['val'], columns=results['str']), os.path.join(sPath, filename[:-4] + '.csv'))

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
