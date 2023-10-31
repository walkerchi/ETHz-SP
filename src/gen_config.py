"""
This file is used to generate config files for experiments

Usage:
>>> python gen_config.py
"""
import argparse 
import torch
import os
import toml

TRAINERS = ["one","multi"]
MODELS = ["GAT","GCN","GraphUNet","MLP","SIGN", "NodeEdgeGNN"]
TRAIN_RATIOS = [0.05,0.1,0.2,0.3,0.5]
SEED   = 123456

def gen_train(path, device):
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        config = {
            "seed":SEED,
            "device": device,
            "task":"train",
            "model":model,
            "dataset":"spherical_shell",
            "d":0.2,
            "use_condense":False,
            "lr":0.01,
            "epochs":1000,
            "train_ratios":TRAIN_RATIOS,
            "test_ratio":0.5,
            "eval_every_eps":5,
        }
        with open(os.path.join(path,f"{model.lower()}.toml"),"w") as f:
            toml.dump(config,f)
        config["use_condense"] = True 
        with open(os.path.join(path,f"{model.lower()}_condense.toml"),"w") as f:
            toml.dump(config,f)

def gen_train_condense(path, device):
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        config = {
            "seed":SEED,
            "device": device,
            "task":"train",
            "force":True,
            "model":model,
            "dataset":"spherical_shell",
            "d":0.2,
            "use_condense":True,
            "lr":0.01,
            "epochs":1000,
            "train_ratios":TRAIN_RATIOS,
            "test_ratio":0.5,
            "eval_every_eps":5,
        }
        with open(os.path.join(path,f"{model.lower()}.toml"),"w") as f:
            toml.dump(config,f)

def gen_scale_variant_test(path):
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        config = {
            "seed" : SEED,
            "task" : "test",
            "model": model,
            "dataset":"spherical_shell",
            "d":0.1,
            "use_condense":False,
            "train_ratios":TRAIN_RATIOS,
        }
        with open(os.path.join(path,f"{model.lower()}.toml"),"w") as f:
            toml.dump(config,f)
        config["use_condense"] = True
        with open(os.path.join(path,f"{model.lower()}_condense.toml"),"w") as f:
            toml.dump(config,f)

def gen_scale_invariant_test(path):
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        config = {
            "seed" : SEED,
            "task" : "test",
            "model": model,
            "dataset":"spherical_shell",
            "d":0.2,
            "use_condense":False,
            "train_ratios":TRAIN_RATIOS,
        }
        with open(os.path.join(path,f"{model.lower()}.toml"),"w") as f:
            toml.dump(config,f)
        config["use_condense"] = True
        with open(os.path.join(path,f"{model.lower()}_condense.toml"),"w") as f:
            toml.dump(config,f)

def gen_multi_train(path, device):
    if not os.path.exists(path):
        os.mkdir(path)
    for use_condense in [True, False]:
        for model in MODELS:
            config = {
                "trainer":"multi",
                "seed":SEED,
                "device": device,
                "task":"train",
                "force":True,
                "model":model,
                "dataset":"truss",
                "use_condense":use_condense,
                "lr":0.01,
                "epochs":1000,
                "train_ratios":TRAIN_RATIOS,
                "test_ratio":0.5,
                "eval_every_eps":5,
            }
            with open(os.path.join(path,f"{model.lower()}{'_condense' if use_condense else ''}.toml"),"w") as f:
                toml.dump(config,f)

def gen_multi_test(path, device):
    if not os.path.exists(path):
        os.mkdir(path)
    for use_condense in [True, False]:
        for model in MODELS:
            config = {
                "trainer":"multi",
                "seed" : SEED,
                "task" : "test",
                "model": model,
                "dataset":"truss",
                "use_condense":use_condense,
                "train_ratios":TRAIN_RATIOS,
            }
            with open(os.path.join(path,f"{model.lower()}{'_condense' if use_condense else ''}.toml"),"w") as f:
                toml.dump(config,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu","cuda"])
    args = parser.parse_args()


    if not os.path.exists("./config"):
        os.mkdir("./config")
    gen_train("./config/train", args.device)
    gen_train_condense("./config/train_condense", args.device)
    gen_scale_variant_test("./config/scale_variant_test")
    gen_scale_invariant_test("./config/scale_invariant_test")
    gen_multi_train("./config/multi_train", "cpu")
    gen_multi_test("./config/multi_test", "cpu")