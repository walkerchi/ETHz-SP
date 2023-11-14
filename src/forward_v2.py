import torch
import torch.nn as nn
from torch_sparse import spmm
import os 
import scipy.sparse
import numpy as np
from gnn import MLP
from linear_elasticity import TrussSolver, partite
from dataset import truss
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse



class EdgeMLP(nn.Module):
    def __init__(self, num_feature, num_hidden=32, num_layers=3, act="relu"):
        super().__init__()
        self.mlp = MLP(num_feature*2, 1, num_hidden, num_layers, act=act)

    def forward(self, x, edge_index, u):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        x_edge = torch.cat([x_src, x_dst], -1)
        edge_weight = self.mlp(x_edge).squeeze() # [n_edge, 1]
        f =      spmm(edge_index, edge_weight, x.shape[0], x.shape[0], u[:, None])[:, 0]
        return f

    def eval_galerkin(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            x_src = x[edge_index[0]]
            x_dst = x[edge_index[1]]
            x_edge = torch.cat([x_src, x_dst], -1)
            edge_weight = self.mlp(x_edge).squeeze() # [n_edge, 1]
            K = scipy.sparse.coo_matrix((edge_weight.numpy(), edge_index.numpy()), shape=(x.shape[0], x.shape[0]))
        return K
    
import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        """
        Apply positional encoding to the input.

        Args:
            x (torch.Tensor): A tensor of shape (..., 3) where ... represents any number of dimensions

        Returns:
            torch.Tensor: The encoded tensor
        """
        # Initialize the list of frequency bands
        frequency_bands = 2.0 ** torch.arange(self.L)
        
        # Reshape x to (..., 1) if it is not already
        n_points = x.shape[0]
        x = x.view(-1, 1)
        
        # Duplicate x for each frequency band (..., L)
        x = x * frequency_bands.view(1, -1)
        
        # Apply sin and cos separately (..., L, 2)
        sin_x = torch.sin(math.pi * x)
        cos_x = torch.cos(math.pi * x)
        
        # Concatenate along the last dimension to interleave sin and cos
        encoded_x = torch.cat((sin_x, cos_x), dim=-1)
        
        # Reshape to the original shape with an additional dimension for the encodings
        encoded_x = encoded_x.reshape(n_points, -1)

        return encoded_x



def deep_galerkin(args):
    mesh = truss.bridge.pratt(n_grid=8, support=2)
    solver = TrussSolver(mesh)
    u      = solver.scipy_solve()     # [n_point, n_dim]
    f      = solver.source_value      # [n_point, n_dim]
    P   = solver.ele2msh_edge_torch   # [n_edge, n_element * n_basis * n_basis]
    x   = solver.points.flatten()
    pos = solver.points[solver.truss] # [n_element, n_basis, n_dim]
    n_element,  n_basis, n_dim = pos.shape
    pos = pos.reshape(n_element, n_basis*n_dim)# [n_element, n_basis * n_dim]
    row, col = solver.K_coo.row, solver.K_coo.col
    
    P   = P.float().to(args.device)
    u   = torch.from_numpy(u).float().view(-1).to(args.device)
    f   = torch.from_numpy(f).float().view(-1).to(args.device)
    x   = torch.from_numpy(x).float().to(args.device).reshape(-1,1)
    pos = torch.from_numpy(pos).float().to(args.device)
    mask= torch.from_numpy(solver.dirichlet_mask).bool().view(-1).to(args.device)
    edges = torch.from_numpy(np.stack([row, col], 0)).long().to(args.device)
    row, col = torch.from_numpy(row), torch.from_numpy(col).to(args.device)

    if args.L is not None:
        pos_enc = PositionalEncoding(args.L)
        x = pos_enc(x)
        model = EdgeMLP(num_feature=args.L*2,  num_hidden=32, num_layers=4, act="tanh")
    else:
        model = EdgeMLP(num_feature=1,  num_hidden=32, num_layers=3, act="tanh")
    model = model.to(args.device)
    loss_fn = nn.SmoothL1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(),  lr=0.002)
    if args.scheduler is None:
        scheduler = None 
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    elif args.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    elif args.scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)
    else:
        raise NotImplementedError
  
    losses = []
    pbar = tqdm(range(args.epoch))
    best_loss, best_weight, best_epoch = 1e10, None, 0
    for ep in pbar:
        optimizer.zero_grad()
        f_  = model(x, edges, u).flatten()
        loss =loss_fn(f,f_)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({"loss": loss.item()})
        if loss.item() < best_loss:
            best_epoch = ep
            best_loss = loss.item()
            best_weight = model.state_dict()
        losses.append(loss.item())
    model.load_state_dict(best_weight)

    model.eval()
    model.cpu()
    x = x.cpu()
    edges = edges.cpu()

    K_  = model.eval_galerkin(x, edges)

    vmin, vmax = min(K_.min(), solver.K_coo.min()), max(K_.max(), solver.K_coo.max())
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].set_title("ground truth")
    ax[1].set_title("prediction")
    ax[0].matshow(solver.K_coo.toarray(), vmin=vmin, vmax=vmax, cmap="jet")
    im = ax[1].matshow(K_.toarray(), vmin=vmin, vmax=vmax, cmap="jet")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size of the colorbar axes
    cbar = fig.colorbar(im, cax=cbar_ax)
    path = os.path.join(".result","forward_v2","K")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,f"L={args.L}.png"))
    fig.savefig(os.path.join(path,f"L={args.L}.pdf"))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title("loss")
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.scatter(best_epoch, best_loss, c="r", marker="*", label="best loss")
    ax.legend()
    path = os.path.join(".result","forward_v2","loss")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,f"L={args.L}.png"))
    fig.savefig(os.path.join(path,f"L={args.L}.pdf"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456)	
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument("--L", type=int,  default=None, help="position encoding length")
    parser.add_argument("-s","--scheduler", type=str, default=None, choices=[None, "exp","step","cos"])
    parser.add_argument("-ep","--epoch", type=int, default=10000)
    args = parser.parse_args()
    deep_galerkin(args)