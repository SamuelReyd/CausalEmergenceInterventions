import matplotlib.pyplot as plt
import numpy as np

def plot_macro_autoprediction(C_M, E_M, title="Macro self-prediction"):
    plt.figure(figsize=(5, 5))
    plt.scatter(C_M, E_M, s=8, alpha=0.35)
    plt.xlabel("V(t)")
    plt.ylabel("V(t')")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.show()

def plot_micro_vs_macro_future(
    C_m,
    E_M,
    feature_names=None,
    I_up=None,
    max_plots=32,
    ncols=3,
    title="Micro vs future macro",
):

    d = C_m.shape[1]
    m = min(d, max_plots)

    nrows = int(np.ceil(m / ncols))

    _, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 1.5 * nrows), sharex="col", sharey="row")

    for j in range(m):
        # ax = plt.subplot(nrows, ncols, j + 1)
        ax = axes.flatten()[j]
        ax.scatter(C_m[:, j], E_M, s=6, alpha=0.3)

        if feature_names is None:
            name = f"feature {j}"
        else:
            name = feature_names[j]

        corr = np.corrcoef(C_m[:, j], E_M)[0, 1]
        sub_title = f"{name}\nr={corr:.2f}"
        if I_up is not None:
            sub_title += f' - I={I_up[j]:.2f}'
        # ax.set_title(sub_title)
        # ax.set_xlabel(name)
        # ax.set_ylabel("V(t')")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()