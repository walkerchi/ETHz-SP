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


class BiLinMLP(nn.Module):
    def __init__(self, n_basis,n_dim, num_hidden, num_layers, act="relu", n_order=2):
        super().__init__()
        """
            B [n_basis*n_dim]->[n_basis*n_dim * num_order]
            D [num_order, num_order]
        """
        self.B_mlp = MLP(n_basis*n_dim, n_basis*n_dim*n_order, num_hidden, num_layers, act=act)
        self.D_triu = torch.nn.Parameter(torch.randn(n_order * (n_order + 1) // 2))
        self.n_order = n_order
        self.n_basis = n_basis
        self.n_dim   = n_dim
    def D(self):
        """
            Returns:
            --------
            D [num_order, num_order]
        """
        matrix = torch.zeros(self.n_order, self.n_order).to(self.D_triu.device)
        matrix[np.triu_indices(self.n_order)] = self.D_triu
        matrix = matrix + matrix.T - torch.diag(matrix.diag())
        return matrix
    def B(self, x):
        """
            Parameters:
            -----------
            x [n_element, n_basis * n_dim]
            Returns:
            -------- 
            B [n_element, n_basis * n_dim,n_order]
        """
        return self.B_mlp(x).reshape(-1, self.n_basis*self.n_dim,  self.n_order)
    def forward(self, x):
        """ B  @ D @ B^T
            Parameters:
            -----------
            x: [n_element, n_basis * n_dim]
            Returns:
            --------
            K: [n_elemen * n_basis * n_basis * n_dim * n_dim]
        """
        B = self.B(x) # [n_element, n_basis * n_dim, n_order]
        D = self.D() # [n_order, n_order]
        K = torch.einsum("nai,ij,nbj->nab", B, D, B)
        K = K.reshape(-1, self.n_basis, self.n_dim, self.n_basis, self.n_dim).permute(0, 1, 3, 2, 4).reshape(-1, self.n_dim * self.n_dim)
        return K
    

class  LinMLP(nn.Module):
    def __init__(self, n_basis,n_dim, num_hidden, num_layers, act="relu"):
        super().__init__()
        self.mlp = MLP(n_basis*n_dim, n_basis*n_basis*n_dim*n_dim, num_hidden, num_layers, act=act)
        self.n_basis = n_basis
        self.n_dim   = n_dim
    def forward(self, x):
        """
            Parameters:
            -----------
            x [n_element, n_basis * n_dim]
            Returns:
            --------
            K [n_element, n_basis * n_basis * n_dim * n_dim]
        """
        K = self.mlp(x).reshape(-1, self.n_dim * self.n_dim)
        return K

def deep_galerkin(args):
    mesh = truss.bridge.pratt(n_grid=8, support=2)
    solver = TrussSolver(mesh)
    u      = solver.scipy_solve()     # [n_point, n_dim]
    f      = solver.source_value      # [n_point, n_dim]
    P   = solver.ele2msh_edge_torch   # [n_edge, n_element * n_basis * n_basis]
    pos = solver.points[solver.truss] # [n_element, n_basis, n_dim]
    n_element,  n_basis, n_dim = pos.shape
    pos = pos.reshape(n_element, n_basis*n_dim)# [n_element, n_basis * n_dim]
    row, col = solver.K_coo.row, solver.K_coo.col
    
    P   = P.float().to(args.device)
    u   = torch.from_numpy(u).float().view(-1).to(args.device)
    f   = torch.from_numpy(f).float().view(-1).to(args.device)
    pos = torch.from_numpy(pos).float().to(args.device)
    mask= torch.from_numpy(solver.dirichlet_mask).bool().view(-1).to(args.device)
    edges = torch.from_numpy(np.stack([row, col], 0)).long().to(args.device)
    row, col = torch.from_numpy(row), torch.from_numpy(col).to(args.device)

    if args.model == "linear":
        model = LinMLP(n_basis, n_dim, 64, 4, act="relu")
    elif args.model == "bilinear":
        model = BiLinMLP(n_basis, n_dim, 64, 4, act="relu")
    else:
        raise NotImplementedError
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(),  lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.85)
    
    losses = []
    epoch = 10000
    pbar = tqdm(range(epoch))
    best_loss, best_weight, best_epoch = 1e10, None, 0
    for ep in pbar:
        optimizer.zero_grad()
        K_local = model(pos) # [n_element* n_basis * n_basis, n_dim * n_dim]
        edata = P @ K_local # [n_edge, n_dim, n_dim]
        f_ = spmm(edges, edata.reshape(-1), u.shape[0], u.shape[0], u[:, None])[:, 0]
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
    P = P.cpu()
    pos = pos.cpu()

    with torch.no_grad():
        K_local = model(pos)
        edata = P @ K_local
    edata = edata.cpu().numpy().ravel()
    
    # K_ = scipy.sparse.coo_matrix(((row,  col), edata), shape=(solver.n_points, solver.n_points))
    K_  = scipy.sparse.coo_matrix((edata, (row, col)), shape=(u.shape[0], u.shape[0]))

    vmin, vmax = min(K_.min(), solver.K_coo.min()), max(K_.max(), solver.K_coo.max())
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].set_title("ground truth")
    ax[1].set_title("prediction")
    ax[0].matshow(solver.K_coo.toarray(), vmin=vmin, vmax=vmax, cmap="jet")
    im = ax[1].matshow(K_.toarray(), vmin=vmin, vmax=vmax, cmap="jet")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size of the colorbar axes
    cbar = fig.colorbar(im, cax=cbar_ax)
    path = os.path.join(".result","forward_v1","K")
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
    path = os.path.join(".result","forward_v1","loss")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,f"{args.model}.png"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456)	
    parser.add_argument('--device', type=str, default= "cpu")
    parser.add_argument("--model", type=str,  default="linear", choices=['linear', 'bilinear'])
    args = parser.parse_args()
    deep_galerkin(args)