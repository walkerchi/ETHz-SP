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
    def __init__(self, num_hidden=32, num_layers=3, act="relu"):
        super().__init__()
        self.mlp = MLP(2, 1, num_hidden, num_layers, act=act)

    def forward(self, x, edge_index, u):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        x_edge = torch.stack([x_src, x_dst], -1)
        edge_weight = self.mlp(x_edge).squeeze() # [n_edge, 1]
        f =      spmm(edge_index, edge_weight, x.shape[0], x.shape[0], u[:, None])[:, 0]
        return f

    def eval_galerkin(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            x_src = x[edge_index[0]]
            x_dst = x[edge_index[1]]
            x_edge = torch.stack([x_src, x_dst], -1)
            edge_weight = self.mlp(x_edge).squeeze() # [n_edge, 1]
            K = scipy.sparse.coo_matrix((edge_weight.numpy(), edge_index.numpy()), shape=(x.shape[0], x.shape[0]))
        return K

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
    x   = torch.from_numpy(x).float().to(args.device)
    pos = torch.from_numpy(pos).float().to(args.device)
    mask= torch.from_numpy(solver.dirichlet_mask).bool().view(-1).to(args.device)
    edges = torch.from_numpy(np.stack([row, col], 0)).long().to(args.device)
    row, col = torch.from_numpy(row), torch.from_numpy(col).to(args.device)

    model = EdgeMLP(num_hidden=32, num_layers=3, act="tanh")
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(),  lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    losses = []
    epoch = 10000
    pbar = tqdm(range(epoch))
    best_loss, best_weight, best_epoch = 1e10, None, 0
    for ep in pbar:
        optimizer.zero_grad()
        f_  = model(x, edges, u).flatten()
        loss = (f-f_).pow(2).mean()
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
    fig.savefig(os.path.join(path,f"{args.model}.png"))

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
    fig.savefig(os.path.join(path,f"{args.model}.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456)	
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument("--model", type=str,  default="linear", choices=['linear', 'bilinear'])
    args = parser.parse_args()
    deep_galerkin(args)