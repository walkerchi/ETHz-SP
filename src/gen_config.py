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
# TRAIN_RATIOS = [0.0, 0.05,0.1,0.2,0.3,0.5]
TRAIN_RATIOS = [0.0, 0.1, 0.3]
SEED   = 123456
PHY_WEIGHT = 1e-2
LR     = 1e-3
EPOCHS = 1000
NUM_HOPS = 8

def gen_train(path, device, physical_loss=None, dataset="spherical_shell", d=0.2, train_ratio=None, loss="weight"):
    if train_ratio is None:
        train_ratio = TRAIN_RATIOS
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        for condense in [None, "static", "nn"]:
           
            if model in ["SIGN"]:
                lr = LR * 0.1
            elif model in ["MLP"]:
                lr = LR * 0.5 
            else:
                lr = LR 

            if model in ["SIGN", "MLP", "GAT"]:
                epochs = EPOCHS * 10
            else:
                epochs = EPOCHS
            config = {
                "seed":SEED,
                "device": device,
                "task":"train",
                "model":model,
                "dataset":dataset,
                "d":d,
                "lr":lr,
                "epochs":epochs,
                "train_ratios":TRAIN_RATIOS,
                "test_ratio":0.5,
                "eval_every_eps":5,
                "physical_weight":PHY_WEIGHT,
                "num_hops":NUM_HOPS,
                "loss":loss,
            }
            if physical_loss is not None:
                config["physical_loss"] = physical_loss
            if condense is not None:
                config["condense"] = condense
            with open(os.path.join(path,f"{model.lower()}{f'_{condense}_condense' if condense is not None else ''}.toml"),"w") as f:
                toml.dump(config,f)

def gen_test(path, dataset="spherical_shell", d=0.2, physical_loss=None, loss="weight"):
    if not os.path.exists(path):
        os.mkdir(path)
    for model in MODELS:
        for condense in [None, "static", "nn"]:
           
            config = {
                "seed" : SEED,
                "task" : "test",
                "model": model,
                "dataset":dataset,
                "d":d,
                "train_ratios":TRAIN_RATIOS,
                "num_hops":NUM_HOPS,
                "loss":loss,
            }
            if condense is not None:
                config["condense"] = condense
            if physical_loss is not None:
                config["physical_loss"] = physical_loss
            with open(os.path.join(path,f"{model.lower()}{f'_{condense}_condense' if condense is not None else ''}.toml"),"w") as f:
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


    # if not os.path.exists("./config"):
    #     os.mkdir("./config")
    # gen_train("./config/train", args.device, d=0.2)
    # gen_train_condense("./config/train_condense", args.device)
    # gen_test("./config/scale_variant_test", d=0.1)
    # gen_test("./config/scale_invariant_test", d=0.2)
    # gen_scale_invariant_test("./config/scale_invariant_test")
    # gen_multi_train("./config/multi_train", "cpu")
    # gen_multi_test("./config/multi_test", "cpu")
    for loss in ["equal", "weight", "auto_weight"]:
        gen_train(f"./config/train_rectangle_{loss}", args.device, physical_loss=None, dataset="rectangle_sin2", d=0.1, loss=loss)
        gen_train(f"./config/train_rectangle_strong_pinn_{loss}", args.device, physical_loss="strong", dataset="rectangle_sin2", d=0.1, loss=loss)
        gen_train(f"./config/train_rectangle_weak_pinn_{loss}", args.device, physical_loss="weak", dataset="rectangle_sin2", d=0.1, loss=loss)

        gen_test(f"./config/invariant_test_rectangle_{loss}", dataset="rectangle_sin2", d=0.1, loss=loss)
        gen_test(f"./config/invariant_test_rectangle_strong_pinn_{loss}",  dataset="rectangle_sin2", d=0.1, physical_loss="strong", loss=loss)
        gen_test(f"./config/invariant_test_rectangle_weak_pinn_{loss}",  dataset="rectangle_sin2", d=0.1, physical_loss="weak", loss=loss)

        gen_test(f"./config/frequency_variant_test_rectangle_{loss}", dataset="rectangle_sin1", d=0.1, loss=loss)
        gen_test(f"./config/frequency_variant_test_rectangle_strong_pinn_{loss}", dataset="rectangle_sin1", d=0.1, physical_loss="strong", loss=loss)
        gen_test(f"./config/frequency_variant_test_rectangle_weak_pinn_{loss}", dataset="rectangle_sin1", d=0.1, physical_loss="weak", loss=loss)

        gen_test(f"./config/boundary_variant_test_rectangle_{loss}", dataset="rectangle_sin2_left+right", d=0.1, loss=loss)
        gen_test(f"./config/boundary_variant_test_rectangle_strong_pinn_{loss}", dataset="rectangle_sin2_left+right", d=0.1, physical_loss="strong", loss=loss)
        gen_test(f"./config/boundary_variant_test_rectangle_weak_pinn_{loss}", dataset="rectangle_sin2_left+right", d=0.1, physical_loss="weak", loss=loss)

        gen_test(f"./config/boundary_left_variant_test_rectangle_{loss}", dataset="rectangle_sin2_left", d=0.1, loss=loss)
        gen_test(f"./config/boundary_left_variant_test_rectangle_strong_pinn_{loss}", dataset="rectangle_sin2_left", d=0.1, physical_loss="strong", loss=loss)
        gen_test(f"./config/boundary_left_variant_test_rectangle_weak_pinn_{loss}", dataset="rectangle_sin2_left", d=0.1, physical_loss="weak", loss=loss)