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
from setup_cifar10 import setup_cifar10

# setup parser
parser = setup_parser()
parser.add_argument('--hamiltonian', action='store_true', help='use Hamiltonian encoder instead of fully-connected')
args = parser.parse_args()

# seed for reproducibility
seed_everything(args.seed)

# setup data
train_loader, val_loader, test_loader = setup_cifar10(args.n_train, args.n_val, args.n_test, args.batch_size, args.data_dir)

# --- Wrap DataLoaders for regression (autoencoder: target = input) ---
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
val_loader   = DataLoader(RegressionWrapper(val_loader.dataset),   batch_size=args.batch_size)
test_loader  = DataLoader(RegressionWrapper(test_loader.dataset),  batch_size=args.batch_size)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CIFAR-10: images are [N, 3, 32, 32]
# Reshaped to TNN layout: [N, 3, 32, 32] -> View -> [N, 96, 32] -> Permute -> [96, N, 32]
# so dim3=32, spatial_dim=32*3=96
dim3        = 32
spatial_dim = 32 * 3  # 96 (channels x height, flattened)

# Create transformation matrix
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
elif args.M == 'data':
    A = torch.empty(0)
    for data, _ in train_loader:
        A = torch.cat((A, data), dim=0)
    A = A.reshape(-1, A.shape[-1]).T
    M, _, _ = torch.linalg.svd(A, full_matrices=False)
    M = M.T.to(dtype=torch.float32, device=device)
    del A
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)

# --- Build encoder and decoder ---
# Decoder is always: tLinearLayer(width->spatial_dim, no activation) solved by VarPro, then reshape
# Encoder is either:
#   FC:          View -> Permute -> tFullyConnected(spatial_dim -> width)
#   Hamiltonian: View -> Permute -> tLinearLayer(spatial_dim -> width) -> tHamiltonianResNet(width, depth)
if args.hamiltonian:
    # Opening layer projects spatial_dim -> width, then Hamiltonian block refines features stably
    encoder = torch.nn.Sequential(
        View((-1, spatial_dim, dim3)),
        Permute((1, 0, 2)),
        tLinearLayer(spatial_dim, args.width, dim3, M=M, activation=torch.nn.Tanh(), bias=args.bias),
        tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                           depth=args.depth, h=args.h_step, activation=torch.nn.Tanh())
    ).to(device)
else:
    encoder = torch.nn.Sequential(
        View((-1, spatial_dim, dim3)),
        Permute((1, 0, 2)),
        tFullyConnected((spatial_dim, args.width), dim3, M=M, activation=torch.nn.Tanh(), bias=args.bias)
    ).to(device)

decoder = torch.nn.Sequential(
    tFullyConnected((args.width, spatial_dim), dim3, M=M, activation=None, bias=True),
    Permute((1, 0, 2)),
    View((-1, 3, 32, 32))
).to(device)

net = AutoencoderWrapper(encoder, decoder).to(device)

# Loss: MSE (pixel-wise reconstruction)
# Optimizer: Adam on encoder only — decoder is updated by VarPro closed-form least squares
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# Logger and output path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
arch = 'hamiltonian' if args.hamiltonian else 'fc'
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', f'tensor_varpro_{arch}', start_time)
filename = f'cifar10_tensor_varpro_{arch}.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info('cifar10_tensor_varpro')
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

# Train
import time as _time
if torch.cuda.is_available():
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    results = train_tnn_varpro_autoencoder(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, n_warmup=args.n_warmup, alpha2=args.weight_decay,
        max_grad_norm=args.max_grad_norm, scheduler=scheduler,
        device=device, logger=logger, sPath=sPath, M=M
    )
    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) / 1000.0
else:
    t0 = _time.perf_counter()
    results = train_tnn_varpro_autoencoder(
        net, criterion, optimizer, train_loader, val_loader, test_loader,
        max_epochs=args.max_epochs, n_warmup=args.n_warmup, alpha2=args.weight_decay,
        max_grad_norm=args.max_grad_norm, scheduler=scheduler,
        device=device, logger=logger, sPath=sPath, M=M
    )
    t1 = _time.perf_counter()
    elapsed = t1 - t0

logger.info('Total Training Time: {:.2f} seconds'.format(elapsed))
torch.save(net.state_dict(), sPath + '/last_net.pt')

results = results[-1] if isinstance(results, tuple) else results
results['args'] = args
pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(
    pd.DataFrame(results['val'], columns=results['str']),
    os.path.join(sPath, filename[:-4] + '.csv')
)

# Visualize reconstructions
import matplotlib.pyplot as plt

x, _ = next(iter(test_loader))
x = x.to(device)
with torch.no_grad():
    x_rec = net(x).cpu()
x = x.cpu()

# CIFAR-10 is normalized with mean=0.5, std=0.5 — denormalize to [0, 1] for display
def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

plt.figure(figsize=(16, 4))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(denorm(x[i]).permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(2, 8, i + 9)
    plt.imshow(denorm(x_rec[i]).permute(1, 2, 0))
    plt.axis('off')
plt.suptitle('Top: original  |  Bottom: reconstruction')
plt.tight_layout()
plt.savefig(sPath + '/reconstructions.png')
plt.show()
