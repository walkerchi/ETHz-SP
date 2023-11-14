import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

colors = plt.cm.tab10.colors
COLORS = {}
MARKERS = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]


def plot_test(test_scores, title, path=None):
    """
        Parameters:
        -----------
            test_scores: #dict[str,float] or dict[str,dict[float,float]] or 
                dict[str,dict[float,dict[str]]]
                The test scores for different models
                
    """
    assert isinstance(test_scores, dict), f"test_scores should be a dict, got {type(test_scores)}"
    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)
    data = {
        "model":[],
        "train_ratio":[],
        "mse":[],
        "use_condense":[]
    }
    for i,(model, loss_per_ratio) in enumerate(test_scores.items()):
        for train_ratio, loss in loss_per_ratio.items():
            data["model"].append(model.split("_")[0])
            data["train_ratio"].append(train_ratio)
            data["mse"].append(loss)
            if model.endswith("_condense"):
                data["use_condense"].append(True)
            else:
                data["use_condense"].append(False)
    df = pd.DataFrame(data)
    sns.lineplot(data=df, x="train_ratio", y="mse", hue="model", style="use_condense" , markers=True, ax=ax)
    # sns.scatterplot(data=df, x="train_ratio", y="mse", hue="model", style="use_condense" , ax=ax, label=None)

    # if isinstance(next(iter(test_scores.values())), dict): # dict[str,dict[float,float]]
    #     for i,(model, score) in enumerate(test_scores.items()):
            
    #         train_ratios, mse_losses = list(score.keys()), list(score.values())

    #         if model.endswith("_condense"):
    #             key = model[:-len("_condense")]
    #             alpha = 0.7
    #             marker = "*"
    #             linestyle = ":"
    #         else:
    #             key = model
    #             alpha = 1.0
    #             marker = "o"
    #             linestyle = "--"
    #         if key in COLORS:
    #             color = COLORS[key]
    #         else:
    #             color = colors[len(COLORS)]
    #             COLORS[key] = color         
            
    #         ax.scatter(train_ratios, mse_losses, label=model, marker=marker, c=color, alpha=alpha)

    #         ax.plot(train_ratios, mse_losses, label=model, linestyle=linestyle, color=color, alpha=0.5)
    # else:
    #     for i, (model, score) in enumerate(test_scores.items()):
    #         if model.endswith("_condense"):
    #             key = model[:-len("_condense")]
    #             alpha = 0.7
    #             marker = "*"
    #         else:
    #             key = model
    #             alpha = 1.0
    #             marker = "o"
    #         if key in COLORS:
    #             color = COLORS[key]
    #         else:
    #             color = colors[len(COLORS)]
    #             COLORS[key] = color
    #         ax.scatter([model], [score], label=model, marker=marker, c=color, alpha=alpha)
    # ax.set_xlabel("train ratio")
    # ax.set_ylabel("mse loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax.set_title(title)

    if path is not None:
        fig.savefig(path)
        fig.savefig(path.replace(".png", ".pdf"))
    else:
        fig.savefig(".result/test_score.png")
        fig.savefig(".result/test_score.pdf")
