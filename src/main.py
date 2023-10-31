import os
import torch 
import numpy as np
import argparse 
import toml
import json
from gnn import OneTrainer, MultiTrainer

from plot import plot_test

def parse_args(parser):
    # hyper parameters
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # for task 
    parser.add_argument('--task', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--force', action='store_true')
    # for trainer
    parser.add_argument('--trainer', type=str, default="multiscale")

    # for dataset
    parser.add_argument('--d', type=float, default=0.2, help="mesh size")
    parser.add_argument('--E', type=float, default=1.0, help="Young's modulus")
    parser.add_argument('--nu', type=float, default=0.4, help="Poisson's ratio")
    parser.add_argument('--a', type=float, default=1.0, help="inner radius")
    parser.add_argument('--b', type=float, default=2.0, help="outer radius")
    parser.add_argument('--p', type=float, default=1.0, help="pressure")
    parser.add_argument('--n_grid', type=int, default=12, help="number of grid")
    parser.add_argument('--n_support', type=int, default=3, help="support of boundary condition")

    # for trainer 
    parser.add_argument("--trainer", type=str, default="one", choices=["one", "multi"])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--train_ratios', nargs="+", default=None)
    parser.add_argument('--test_ratio', type=float, default=0.5)
    parser.add_argument('--eval_every_eps', type=int, default=5)


    parser.add_argument('--dataset', type=str, default='spherical_shell')
    # for general model
    parser.add_argument('--use_edata', action='store_true')
    parser.add_argument('--use_condense', action='store_true')
    parser.add_argument('--condense_model', type=str, default="gin-2", choices=['gin-2'] )
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)

    # for GAT
    parser.add_argument('--num_heads', type=int, default=4)

    # for SIGN
    parser.add_argument('--num_hops', type=int, default=3)

    # for GraphUNet
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--pool_ratios', type=float, default=2)
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--conv', type=str, default='GCNConv')
    parser.add_argument('--pool', type=str, default='TopKPooling')

    # for result
    args = parser.parse_args()
    return args

import argparse
import toml
import json

class FileParser:
    def __init__(self, filename = None):
        assert filename.endswith(('.toml', '.json', '.yaml')), f"File type {filename} is not supported"
        if filename is not None:
            with  open(filename) as f:
                if filename.endswith('.json'):
                    self.args = json.load(f)
                else:
                    self.args = toml.load(f)
        else: 
            self.args = {}

    def add_argument(self, *args, **kwargs):
        for arg in args:
            if arg.startswith('--'):
                arg = arg[2:]
                
                if 'action' in kwargs:
                    default = False if kwargs['action'] == 'store_true' else None
                elif 'default' in kwargs:
                    default = kwargs['default']
                else:
                    raise Exception(f"Argument {arg} should have default value")
                self.args[arg] = self.args.get(arg, default)
          
            
        if "choices" in kwargs:
            assert self.args[arg] in kwargs["choices"], f"Argument {arg} should be in {kwargs['choices']}, but got {self.args[arg]}"
        if "type" in kwargs:
            self.args[arg] = kwargs["type"](self.args[arg])
    
    def parse_args(self):
        x = argparse.Namespace()
        for key, value in self.args.items():
            if value is not None:
                x.__dict__[key] = value
        return x
        
class DictParser(FileParser):
    def __init__(self, **kwargs):
        super().__init__(None)
        self.args = kwargs if kwargs is not None else {}

def parse_cmd():
    parser = argparse.ArgumentParser()
    return parse_args(parser)

def parse_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default=None)
    parser.add_argument('-f', "--folder", type=str, default=None)
    parser.add_argument('-t', "--task", type=str, default=None, choices=["all_infer"])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.config is not None:
        arg = parse_args(FileParser(args.config))
        if args.force:
            arg.force = True
        return arg
    elif args.folder is not None:
        l_arg = []
        for filename in os.listdir(args.folder):
            filename = os.path.join(args.folder, filename)
            arg = parse_args(FileParser(filename))
            l_arg.append(arg)
            if args.force:
                arg.force = True
        return l_arg
    # elif args.task == "all_infer":
    #    assert args.folder is None and args.config is None
    #    return parse_args(DictParser(task="all_infer"))


if __name__ == '__main__':
    def manual_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
      
    
    def run_trainer(arg):
        manual_seed(arg.seed)
        Trainer = {
            "one": OneTrainer,
            "multi": MultiTrainer,
        }[arg.trainer]
        
        if arg.task == "train":
            trainer = Trainer(arg)
            if arg.train_ratios is not None:
                for train_ratio in arg.train_ratios:
                    print(f"train_ratio: {train_ratio}")
                    arg.train_ratio = train_ratio
                    trainer = Trainer(arg)
                    if not os.path.exists(trainer.model_weight_path) or not os.path.exists(trainer.loss_image_path) or arg.force:
                        trainer.fit()
            else:
                if not os.path.exists(trainer.model_weight_path) or not os.path.exists(trainer.loss_image_path) or args.force:
                    trainer.fit()
        elif arg.task == "test":
            if arg.train_ratios is not None:
                result = {}
                for train_ratio in arg.train_ratios:
                    print(f"train_ratio : {train_ratio}")
                    arg.train_ratio = train_ratio
                    mse = Trainer(arg).test()
                    result[train_ratio] = mse
                return result
            else:
                return Trainer(arg).test()
        else:
            raise NotImplementedError()
        
    
    
    args = parse_file()
    
    if isinstance(args, (list, tuple)):
        if args[0].task == "test":
            results = {}
            for i,arg in enumerate(args):
                assert arg.train_ratio + arg.test_ratio <= 1.0, f"train_ratio + test_ratio should be less than 1.0"
                assert arg.task == "test", f"task should be test for folder of test, but got {arg.task}"
                print(f"Configurations {i+1}/{len(args)}")
                mse = run_trainer(arg)
                if arg.use_condense:
                    results[f"{arg.model}_condense"] = mse
                else:
                    results[arg.model] = mse
            plot_test(results, f"test on {args[0].dataset} with characteristic length {args[0].d}", f"./.result/{arg.trainer}_test_score_{args[0].d}.png")
        else:
            for i,arg in enumerate(args):
                assert arg.train_ratio + arg.test_ratio <= 1.0, f"train_ratio + test_ratio should be less than 1.0"
                print(f"Configurations {i+1}/{len(args)}")
                run_trainer(arg)
    else:
        assert args.train_ratio + args.test_ratio <= 1.0, f"train_ratio + test_ratio should be less than 1.0"
    
        run_trainer(args)
