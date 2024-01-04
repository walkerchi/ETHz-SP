import torch
import numpy as np
import argparse
import os  
from tqdm import tqdm
from main import OneTrainer, MultiTrainer ,FileParser, parse_args
import matplotlib.pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None)   
    parser.add_argument("-s", "--show", action="store_true", default=False)
    # parser.add_argument("-f" , "--folder", type=str, default=None)
    # parser.add_argument("-i","--index", type=int, default=0, help="index of the train ratio")

    args = parser.parse_args()
    is_show = args.show
    name = os.path.basename(args.config).split(".")[0]
    args = parse_args(FileParser(args.config))
    

    if args.train_ratios is None:
        args.train_ratios = [args.train_ratio]
    for index in tqdm(range(len(args.train_ratios))):
        if args.train_ratios is not None:
            args.train_ratio = args.train_ratios[index]
       
        trainer = {
            "one":OneTrainer,
            "multi":MultiTrainer,
        }[args.trainer](args)
        trainer.load()
        with torch.no_grad():
            if isinstance(trainer, MultiTrainer):
                graph = trainer.test_graph
                graph_i = trainer.test_graph_i
                graph_b2i = trainer.test_graph_b2i
                dataset = trainer.test_dataset
            else:
                graph  = trainer.graph
                graph_i = trainer.graph_i
                graph_b2i = trainer.graph_b2i
                dataset = trainer.dataset
            u_pred = trainer.predict(graph=graph, graph_i=graph_i, graph_b2i=graph_b2i).cpu().numpy()
            
            if args.condense is not None:
                u_global = np.zeros_like(graph.n_pos)
                u_global[~graph.n_dirichlet_mask] += u_pred 
                u_global[graph.n_dirichlet_mask] += graph_b2i.n_displacement_u.cpu().numpy()
            else:
                u_global = u_pred

            loss = trainer.test()

            print(f"loss: {loss}")

        u_gt = graph.n_displacement.cpu().numpy()
        # breakpoint()
        dataset.solver.plot(ux=u_gt[:,0], ux_pred=u_global[:,0], uy=u_gt[:,1], uy_pred=u_global[:,1], show=False)
        if is_show:
            plt.show()
        else:
            path = os.path.join(".result","vis_predict")
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path,f"{args.train_ratio:2.1f}_{args.dataset}_{args.loss}_"+name + ".png")
            plt.savefig(path)

