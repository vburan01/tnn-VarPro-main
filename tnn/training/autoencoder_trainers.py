# ============================================================
# TNN Autoencoder Trainers
# Implements Variable Projection (VarPro) training for tensor
# neural network (TNN) autoencoders using the M-product algebra.
#
# Usage:
#   This file is imported by examples/mnist/ex_mnist_tensor_varpro_regression.py.
#   It is not meant to be run directly. The main entry point is
#   train_tnn_varpro_autoencoder(...).
#
# Key idea (VarPro):
#   Split the autoencoder into a nonlinear encoder and a linear
#   decoder. Instead of using gradient descent for the decoder's
#   last linear layer, solve for it analytically (closed-form
#   least squares) using all training data at once. Only the
#   encoder is updated by gradient descent.
# ============================================================

import torch
from copy import deepcopy
import time
from tnn.training import optimizer_parameters, parameters_norm
from tnn.training.batch_train import test
from . import batch_train
from tnn.tensor_utils.modek_product import modek_product


# ------------------------------------------------------------
# AutoencoderWrapper
# A thin nn.Module that holds encoder and decoder as named
# submodules so that net.encoder and net.decoder work cleanly,
# and net(x) = decoder(encoder(x)).
# ------------------------------------------------------------
class AutoencoderWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ------------------------------------------------------------
# tensor_least_squares_tproduct
#
# Solves the slice-wise linear regression problem in the
# M-transform domain:
#
#   For each frequency slice k:  W_k @ A_k  ≈  B_k
#
# where:
#   A: [bottleneck, n_samples, n3]  — encoder outputs (transform domain)
#   B: [output_dim, n_samples, n3]  — reconstruction targets (transform domain)
#
# If with_bias=True, a row of ones is appended to A before solving,
# so that bias is solved jointly:
#
#   [W_k | b_k] @ [A_k; 1^T]  ≈  B_k
#
# The last column of the solution becomes the bias term.
# This is equivalent to standard affine least squares at zero
# extra cost.
#
# Returns:
#   W_M:    [output_dim, bottleneck, n3]  — weight tensor (transform domain)
#   bias_M: [output_dim, 1, n3]           — bias tensor (transform domain), or None
# ------------------------------------------------------------
def tensor_least_squares_tproduct(A, B, with_bias=False, alpha2=0.0):
    """
    Solves the regularized inner problem (eq. 2.7 from the VarPro paper) slice-wise
    in the M-transform domain:

        W(θ) = arg min_{W}  (1/2n)||WA - B||_F^2 + (alpha2/2)||W||_F^2

    The Tikhonov (ridge) regularization is handled via an augmented least squares
    system — no matrix inversion needed:

        min_{Wk}  ||[Ak^T          ] Wk^T  -  [Bk^T]||_F^2
                  ||[sqrt(n*alpha2)I]           [0   ]||

    This is equivalent to (2.7) per frequency slice k.
    The bias column (if with_bias=True) is NOT regularized, matching standard practice.
    """
    bottleneck, n_samples, n3 = A.shape
    output_dim, n_samples2, n3b = B.shape
    assert n_samples == n_samples2 and n3 == n3b, "A and B must have matching batch and dim3 sizes"

    if with_bias:
        # Augment A with a row of ones: shape becomes [bottleneck+1, n_samples, n3]
        ones = torch.ones(1, n_samples, n3, dtype=A.dtype, device=A.device)
        A_aug = torch.cat([A, ones], dim=0)
        aug_dim = bottleneck + 1
    else:
        A_aug = A
        aug_dim = bottleneck

    X_aug = torch.zeros((output_dim, aug_dim, n3), dtype=A.dtype, device=A.device)

    # Regularization rows: sqrt(n * alpha2) * I on the weight columns only (not bias)
    # Scaled by n_samples so alpha2 has the same meaning regardless of dataset size
    if alpha2 > 0.0:
        reg_scale = (n_samples * alpha2) ** 0.5
        # [bottleneck, aug_dim] — identity on weight columns, zeros on bias column (if any)
        reg_rows = torch.zeros(bottleneck, aug_dim, dtype=A.dtype, device=A.device)
        reg_rows[:bottleneck, :bottleneck] = reg_scale * torch.eye(bottleneck, dtype=A.dtype, device=A.device)
        reg_rhs = torch.zeros(bottleneck, output_dim, dtype=A.dtype, device=A.device)

    for k in range(n3):
        Ak = A_aug[:, :, k]  # [aug_dim, n_samples]
        Bk = B[:, :, k]      # [output_dim, n_samples]

        if alpha2 > 0.0:
            # Augmented system encodes the Tikhonov penalty as extra rows:
            #   [Ak^T          ] Wk^T  =  [Bk^T]
            #   [sqrt(n*a2) * I]           [0   ]
            Ak_reg = torch.cat([Ak.T, reg_rows], dim=0)        # [n_samples + bottleneck, aug_dim]
            Bk_reg = torch.cat([Bk.T, reg_rhs],  dim=0)        # [n_samples + bottleneck, output_dim]
            Wk = torch.linalg.lstsq(Ak_reg, Bk_reg).solution.T # [output_dim, aug_dim]
        else:
            Wk = torch.linalg.lstsq(Ak.T, Bk.T).solution.T    # [output_dim, aug_dim]

        X_aug[:, :, k] = Wk

    if with_bias:
        # Split solution: first bottleneck columns = weight, last column = bias
        return X_aug[:, :bottleneck, :], X_aug[:, bottleneck:, :]
    return X_aug, None


# ------------------------------------------------------------
# train_one_epoch_varpro
#
# Runs one full VarPro epoch in two phases:
#
# Phase 1 — Decoder update (closed-form, globally optimal):
#   1. Run encoder on all training data; collect all latent
#      codes z_all (no gradients needed, detached).
#   2. Pass z_all through all decoder layers except the last.
#   3. Apply M-transform to the intermediate representation
#      and the targets.
#   4. Solve the slice-wise least squares problem to get the
#      globally optimal weight (and bias) for the last layer.
#   5. Write the solution directly into last_layer.weight
#      (and last_layer.bias if used).
#
# Phase 2 — Encoder update (gradient descent):
#   With the decoder's last layer now fixed at its optimal
#   value, run a standard forward/backward pass through the
#   full autoencoder and update the encoder with Adam.
#
# This alternation ensures the decoder is always at its
# global optimum conditioned on the current encoder.
# ------------------------------------------------------------
def train_one_epoch_varpro(net, criterion, optimizer, train_loader, M, device=None, dtype=None, regularizer=None, alpha2=0.0, max_grad_norm=1.0):
    factory_kwargs = {'device': device, 'dtype': dtype}
    net.train()
    criterion.reduction = 'mean'  # reset after any prior test() call which sets 'sum'

    # ---- Phase 1: collect all encoder outputs (no grad) ----
    zs = []
    datas = []
    for data, target in train_loader:
        data = data.to(**factory_kwargs)  # keep as [N, 1, 28, 28] — let encoder handle its own reshaping
        z = net.encoder(data)             # encoder does View+Permute internally → [width, N, 28]
        # Convert data to TNN layout separately for the least-squares target
        data_tnn = data.squeeze(1).permute(1, 0, 2)  # [N, 1, 28, 28] → [28, N, 28]
        zs.append(z.detach())
        datas.append(data_tnn.detach())
    # Concatenate along sample dimension (dim=1 in TNN layout)
    z_all = torch.cat(zs, dim=1)       # [width, total_N, dim3]
    data_all = torch.cat(datas, dim=1) # [28, total_N, 28]

    # ---- Phase 1: propagate through intermediate decoder layers ----
    decoder = net.decoder[0]   # tFullyConnected module
    layers = decoder.layers    # list of tLinearLayer objects inside tFullyConnected
    x = z_all
    for i in range(len(layers) - 1):
        # Apply all layers except the last (these stay fixed, updated only during warmup)
        x = layers[i](x, M)
    # x is now the input to the last decoder layer: [28, total_N, 28]

    last_layer = layers[-1]
    assert last_layer.activation is None, (
        "VarPro requires the last decoder layer to have no activation. "
        f"Got activation={last_layer.activation}"
    )
    use_bias = last_layer.bias is not None

    # ---- Phase 1: solve least squares in M-transform domain ----
    # Apply M-transform (e.g. DCT) to work in the frequency domain
    x_M    = modek_product(x,        M)  # [bottleneck, total_N, n3]  — encoder features
    data_M = modek_product(data_all, M)  # [28, total_N, n3]           — targets

    W_M, bias_M = tensor_least_squares_tproduct(x_M, data_M, with_bias=use_bias, alpha2=alpha2)

    # Transform solution back to spatial domain
    W = modek_product(W_M, M.T)
    if W.shape != last_layer.weight.shape:
        raise RuntimeError(
            f"VarPro weight shape mismatch: got {W.shape}, expected {last_layer.weight.shape}."
        )
    last_layer.weight.data.copy_(W)  # overwrite decoder's last layer weight
    if use_bias:
        b = modek_product(bias_M, M.T)  # [output_dim, 1, dim3]
        last_layer.bias.data.copy_(b)   # overwrite decoder's last layer bias

    # ---- Phase 2: encoder gradient update ----
    running_loss = 0
    num_samples = 0
    for data, target in train_loader:
        data = data.to(**factory_kwargs)  # keep as [N, 1, 28, 28]

        z = net.encoder(data)       # encoder handles View+Permute internally → [width, N, 28]
        x_rec = net.decoder(z)      # decoder includes Permute+View → [N, 1, 28, 28]

        loss = criterion(x_rec, data)
        if regularizer is not None:
            loss = loss + regularizer(net.encoder)

        optimizer.zero_grad()
        loss.backward()
        # Clip encoder gradients to prevent divergence (especially without warmup)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.encoder.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        num_samples += data.size(0)

    # Return (obj, loss, acc) tuple matching batch_train.train_one_epoch format
    # acc=0.0 since this is regression (no class predictions)
    avg_loss = running_loss / num_samples
    return avg_loss, avg_loss, 0.0


# ------------------------------------------------------------
# train_tnn_varpro_autoencoder
#
# Full training loop for the VarPro TNN autoencoder.
# Handles two phases automatically:
#
# Warm-up phase (epochs 0 .. n_warmup-1):
#   All parameters (encoder + decoder) are trained jointly
#   with standard Adam gradient descent. This gives the encoder
#   a non-degenerate starting point before VarPro begins.
#   Without warm-up, deep networks can collapse to a zero-output
#   fixed point where VarPro has nothing useful to work with.
#
# VarPro phase (epochs n_warmup .. max_epochs-1):
#   Alternates between:
#     - closed-form optimal decoder update (train_one_epoch_varpro)
#     - encoder gradient update with Adam (encoder optimizer only)
#
# Logging, result tracking, and best-model saving follow the
# same conventions as batch_train.train().
# ------------------------------------------------------------
def train_tnn_varpro_autoencoder(
    net,
    criterion,
    optimizer,           # encoder-only Adam optimizer (used during VarPro phase)
    train_loader,
    val_loader=None,
    test_loader=None,
    scheduler=None,      # LR scheduler tied to encoder optimizer
    regularizer=None,
    max_epochs=10,
    n_warmup=5,          # number of warm-up epochs using standard GD on all params
    alpha2=0.0,          # L2 regularization on decoder W (eq. 2.7); solved in closed form
    max_grad_norm=1.0,   # gradient clipping for encoder update; None to disable
    device=None,
    dtype=None,
    logger=None,
    sPath='tmp/',
    M=None               # M-transform matrix (e.g. DCT matrix)
):
    factory_kwargs = {'device': device, 'dtype': dtype}

    # Logging/results setup — mirrors batch_train.train() format
    keys, opt_params = optimizer_parameters(optimizer)
    param_norm, grad_norm = parameters_norm(net.encoder)

    results = {
        'headers': ('',) * (4 + len(keys)) + ('running', '', '') + ('train', '') + ('valid', ''),
        'str': ('epoch',) + keys + ('|params|', '|grad|', 'time') +
               ('obj', 'loss', 'acc', 'loss', 'acc', 'loss', 'acc'),
        'frmt': '{:<15d}' + len(keys) * '{:<15.4e}' + '{:<15.4e}{:<15.4e}{:<15.2f}' +
                '{:<15.4e}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}',
        'val': None,
        'best_val_loss': torch.tensor(float('inf')).item(),
        'best_val_loss_epoch': -1,
        'best_val_acc': 0.0,
        'best_val_acc_epoch': -1,
        'total_time': 0.0
    }
    torch.save(net.encoder.state_dict(), sPath + '/best_val_loss_encoder.pt')
    torch.save(net.decoder.state_dict(), sPath + '/best_val_loss_decoder.pt')

    # Record initial performance before any training
    train_out2 = test(net, criterion, train_loader, **factory_kwargs)
    test_out = test(net, criterion, test_loader, **factory_kwargs) if test_loader is not None else (0, 0)
    his = [-1] + opt_params + [param_norm, grad_norm, 0, 0, 0, 0] + [*train_out2] + [*test_out]
    results['val'] = torch.tensor(his).view(1, -1)

    if logger is not None:
        logger.info((len(results['headers']) * '{:<15s}').format(*results['headers']))
        logger.info((len(results['str']) * '{:<15s}').format(*results['str']))
        logger.info(results['frmt'].format(*his))

    # Warm-up optimizer: trains ALL parameters (encoder + decoder) jointly.
    # Uses the same LR as the VarPro encoder optimizer for consistency.
    warmup_optimizer = torch.optim.Adam(
        list(net.encoder.parameters()) + list(net.decoder.parameters()),
        lr=optimizer.param_groups[0]['lr']
    )

    total_start = time.time()
    for epoch in range(max_epochs):
        start = time.time()

        if epoch < n_warmup:
            # --- Warm-up: standard gradient descent on all parameters ---
            if logger is not None and epoch == 0:
                logger.info(f'Starting warm-up phase ({n_warmup} epochs, standard GD for all params)')
            train_out = batch_train.train_one_epoch(
                net, criterion, warmup_optimizer, train_loader,
                regularizer=regularizer, device=device, dtype=dtype
            )
        else:
            # --- VarPro: alternating decoder least-squares + encoder GD ---
            if logger is not None and epoch == n_warmup:
                logger.info(f'Switching to VarPro alternating minimization (epoch {epoch})')
            train_out = train_one_epoch_varpro(
                net, criterion, optimizer, train_loader, M,
                device=device, dtype=dtype, regularizer=regularizer,
                alpha2=alpha2, max_grad_norm=max_grad_norm
            )

        end = time.time()

        # Evaluate on full train and validation sets
        train_out2 = test(net, criterion, train_loader, **factory_kwargs)
        val_out = test(net, criterion, val_loader, **factory_kwargs) if val_loader is not None else (0, 0)

        # Track weight and gradient norms (encoder only during VarPro phase)
        param_norm, grad_norm = parameters_norm(net.encoder)

        # Build log row; pad/trim if column count differs (can happen when
        # the encoder-only optimizer has fewer param groups than initial row)
        his = [epoch]
        _, opt_params = optimizer_parameters(optimizer)
        his += opt_params
        # Insert running stats (obj, loss, acc) to match batch_train column layout
        his += [param_norm, grad_norm, end - start] + list(train_out)
        his += [*train_out2] + [*val_out]
        expected_cols = results['val'].shape[1]
        if len(his) != expected_cols:
            if len(his) < expected_cols:
                his = his + [float('nan')] * (expected_cols - len(his))
            else:
                his = his[:expected_cols]
        results['val'] = torch.cat((results['val'], torch.tensor(his).view(1, -1)), dim=0)

        # Save best model checkpoint based on validation loss
        if val_out[0] <= results['best_val_loss']:
            results['best_val_loss'] = deepcopy(val_out[0])
            results['best_val_loss_epoch'] = epoch
            torch.save(net.encoder.state_dict(), sPath + '/best_val_loss_encoder.pt')
            torch.save(net.decoder.state_dict(), sPath + '/best_val_loss_decoder.pt')

        if logger is not None:
            logger.info(results['frmt'].format(*his))

        # Only step the LR scheduler during VarPro phase — the scheduler is
        # tied to the encoder optimizer which is not used during warm-up.
        if scheduler is not None and epoch >= n_warmup:
            scheduler.step()

    total_end = time.time()
    results['total_time'] = total_end - total_start

    # Final test set evaluation
    if test_loader is not None:
        test_out = test(net, criterion, test_loader, **factory_kwargs)
        if logger is not None:
            logger.info('Test performance:')
            logger.info('loss = {:<15.4e}'.format(test_out[0]))
            logger.info('accuracy = {:<15.4f}'.format(test_out[1]))

    return net.encoder, net.decoder, results
