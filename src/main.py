"""
Regime x macro x prior sweep of Psi_boids with saturated macro features.

Hypothesis under test
---------------------
Every macro is saturated so that "no flocking" states map to a single
constant value. Therefore, under the uniform prior (which produces
"no flocking" states by construction), I(C_M, E_M) ~ 0 and per-boid
MIs ~ 0, so Psi ~ 0.

Under the trajectory prior, we report what happens — the question of
whether Psi is positive depends on whether each macro carries
collective info that no single boid can replicate (synergy) or
redundant info already in each boid.

Usage
-----
    python run_psi_macro_sweep.py

Output
------
One row per (regime, macro, prior), columns: mean_macro, I_macro,
sum_upward, Psi.
"""

import time
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from flocking import pi
from flocking_observables import (
    observe_polarization_sat,
    observe_pairwise_factor,
)
from flocking_pipeline import (
    build_pairs,
    sample_trajectory_confounder
)
from psi import psi_score

from plot_distributions import plot_macro_autoprediction, plot_micro_vs_macro_future

from confounder_simulations import HP_ATTRACTOR_DEFAULT, HP_WIND_DEFAULT, make_sweep_confounder
# from render_simulation import plot_simulation


HP_BASE = dict(
    max_speed=7,
    view_radius=25,
    minimum_separation=10,
    n_steps=150, #80,
    n_boids=10,
    seed=None,
    confounder=False,
    confounder_type=None,
)

REGIMES = {
    "tight":    dict(alignment_factor=15 * pi / 180, separation_factor=13 * pi / 180,
                     cohesion_factor=15 * pi / 180, speed=15,
                     noise=0.01),
    # "moderate": dict(alignment_factor=5 * pi / 180,  separation_factor=5  * pi / 180,
                    #  cohesion_factor=5  * pi / 180, noise=0.05),
    # "loose":    dict(alignment_factor=2 * pi / 180,  separation_factor=2  * pi / 180,
    #                  cohesion_factor=2  * pi / 180, noise=0.10),
    "attractor": HP_ATTRACTOR_DEFAULT | {"max_speed":15},
    "attractor-loose": HP_ATTRACTOR_DEFAULT | {"max_speed":15, "attractor_alpha": .05,  "attractor_noise": 5.0},
    "wind": HP_WIND_DEFAULT | {"max_speed":15},
    "wind-loose": HP_WIND_DEFAULT | {"max_speed":15, "wind_turn_speed": .2 * pi / 180,  "wind_noise": 5 * pi / 180},
}

MIN_SEP = HP_BASE["minimum_separation"]

# Each macro is a callable: (X[n_samples, n_boids*3], n_boids) -> macro array.
# All macros saturate to ~0 under non-flocking inputs.
MACROS = {
    "polarization":     lambda X, n: observe_polarization_sat(X, n, low=0.5, high=1.0),
    "distance": lambda X, n: observe_pairwise_factor(X, n, min_sep=MIN_SEP,
                                                         fact0=0.0, fact1=25.0),
}

PRIORS = [
    "traj", 
    "uniform",
    "noise",
    "confounder"
    ]

LAG = 5
INIT_LAG = 50
N_SAMPLES_REF = 5_000_000
N_SAMPLES = 1_000
N_SURROGATES = 5
K = 20
N_REPEAT = 10
N_INTERVENTIONS = 4
NOISE_INTERVENTIONS = .15

def show_line(regime_name, macro_name, prior, results):
    E_M = np.array([res["E_M"] for res in results])
    C_M = np.array([res["E_M"] for res in results])
    psi = [res["psi"] for res in results]
    sum_up = [res["sum_up"] for res in results]
    I_macro = [res["I_macro"] for res in results]

    saturated = ((E_M == 0) | (E_M == 1) | (C_M == 0) | (C_M == 1)).mean()
    print(
        f"{regime_name:<9} {macro_name:<13} {prior:<15} "
        f"{float(np.mean(C_M)):>7.2f} {float(np.std(C_M)):>7.2f} {saturated:>10.2f} | "
        f"{np.mean(I_macro):>6.2f}±{np.std(I_macro):>5.2f} "
        f"{np.mean(sum_up):>6.2f}±{np.std(sum_up):>5.2f} "
        f"{np.mean(psi):>6.2f}±{np.std(psi):>5.2f}"
    )

def non_valid_combinaison(macro_name, regime_name):
    return (
    (macro_name == "polarization" and "attractor" in regime_name) or 
    (macro_name == "distance" and "wind" in regime_name)
)

def make_figure(regime_name):
    from render_simulation import plot_boids
    from flocking_pipeline import make_interventions_noise, make_initial_conditions_uniform_boids
    hp = HP_BASE | REGIMES[regime_name]
    X = np.load(f"data/{regime_name}.npy")
    traj_flat = X.reshape(np.prod(X.shape[:2]), np.prod(X.shape[2:]))
    uniforms = make_initial_conditions_uniform_boids(traj_flat, 1).reshape(10,3)
    run = 3
    t = 40
    boids = X[run, t]
    np.random.seed(1) #0 ok
    noisies = make_interventions_noise(traj_flat.copy(), hp, np.prod(X.shape[:2]), N_INTERVENTIONS, NOISE_INTERVENTIONS)
    noisies = noisies.reshape(X.shape)
    noisy = noisies[run,t]
    plot_boids(boids, savepath="figures/observations.pdf")
    plot_boids(noisy, savepath="figures/perturbations.pdf")
    plot_boids(uniforms, savepath="figures/uniforms.pdf")

def make_noise_sweep(regime_name, macro_name, max_n=None, max_factor=1, n_values=10):
    """
    Sweep the perturbation distribution over (n_interventions, noise_factor)
    and record Psi at each grid point.

    Rows index n_interventions in [0, max_n]; columns index noise_factor in
    [0, max_factor]. 
    """

    regime_params = REGIMES[regime_name]
    hp = {**HP_BASE, **regime_params}
    n_boids = hp["n_boids"]
    if max_n is None:
        max_n = 3*n_boids
    macro_fn = MACROS[macro_name]
    results = np.zeros((n_values,n_values))
    for i, n_interventions in enumerate(tqdm(np.linspace(0, max_n,n_values,dtype=int))):
        for j, noise_factor in enumerate(np.linspace(0, max_factor, n_values)):
            X, _, _ = build_pairs(
                hp, "noise", LAG, INIT_LAG, N_SAMPLES_REF, N_SAMPLES, 
                n_interventions=n_interventions, noise_factor=noise_factor,
                path=f"data/{regime_name}.npy"
                )
            C_M = macro_fn(X[:, 0], n_boids)
            E_M = macro_fn(X[:, 1], n_boids)
            psi, _, _ = psi_score(
                X[:, 0], C_M, E_M, n_boids,
                k=K, n_surrogates=N_SURROGATES,
            )
            results[i,j] = psi
    np.save(f"data/{macro_name}_{regime_name}_sweep.npy", results)

def set_axes_size(ax, w, h):
    """Force the axes (plot area inside ticks) to exactly w × h inches."""
    fig = ax.get_figure()
    fig.canvas.draw()  # forces layout computation
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    ax_bbox = ax.get_window_extent(renderer)
    
    # Margins around the axes in inches
    dpi = fig.dpi
    left   = (ax_bbox.x0 - tight.x0) / dpi
    right  = (tight.x1 - ax_bbox.x1) / dpi
    bottom = (ax_bbox.y0 - tight.y0) / dpi
    top    = (tight.y1 - ax_bbox.y1) / dpi
    
    fig.set_size_inches(left + w + right, bottom + h + top)

def set_axes_size(ax, w, h):
    fig = ax.get_figure()
    fig.tight_layout()  # compute margins first
    # subplotpars give margins as fractions of figure size
    sp = fig.subplotpars
    # invert: figsize = axsize / fraction
    fig.set_size_inches(w / (sp.right - sp.left), h / (sp.top - sp.bottom))
    fig.tight_layout()  # re-run to clean up after resize

def render_noise_sweep(regime_name, macro_name, max_n=None, max_factor=None,
                       n_values=10, show_bar=True, show_legend=True,
                       axes_w=2.0, axes_h=2.0):  # <-- control plot area here
    sweep_data = np.load(f"data/{macro_name}_{regime_name}_sweep.npy")
    fig, ax = plt.subplots()
    hp = {**HP_BASE, **REGIMES[regime_name]}
    if max_n is None:
        max_n = 3 * hp["n_boids"]
    if max_factor is None:
        max_factor = 1

    x_labels = np.linspace(0, max_n, n_values, dtype=int)
    y_labels = np.linspace(0, max_factor, n_values).round(2)

    im = ax.imshow(sweep_data.T, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.linspace(0, sweep_data.shape[1]-1, len(x_labels)), x_labels)
    ax.set_xlabel("# variables")

    if show_legend:
        ax.set_ylabel("amplitude")
        ax.set_yticks(np.linspace(0, sweep_data.shape[0]-1, len(y_labels)), y_labels)
    else:
        ax.set_yticks([])

    if show_bar:
        fig.colorbar(im, ax=ax)

    set_axes_size(ax, axes_w, axes_h)
    plt.savefig(f"figures/{macro_name}_{regime_name}_sweep.pdf", bbox_inches='tight')

def get_results_micro(prior, regime_name):
    hp = {**HP_BASE, **REGIMES[regime_name]}
    results = []
    for i in range(N_REPEAT):
        X, _, _ = build_pairs(
            hp|{"seed":i}, prior, LAG, INIT_LAG, N_SAMPLES_REF, N_SAMPLES, 
            n_interventions=N_INTERVENTIONS, noise_factor=NOISE_INTERVENTIONS,
            path=f"data/{regime_name}.npy")
        C_m = X[:, 0]
        E_m = X[:, 1]
        results.append({"X": X, "C_m": C_m, "E_m":E_m}) 
    return results

def get_results_macro(prior, regime_name, macro_name, results=None, compute_psi=True):
    if results is None: 
        results = get_results_micro(prior, regime_name)
    if non_valid_combinaison(macro_name, regime_name):
        return
    macro_fn = MACROS[macro_name]
    hp = {**HP_BASE, **REGIMES[regime_name]}
    n_boids = hp["n_boids"]
    for res in results:
        res["C_M"] = macro_fn(res["C_m"], n_boids)
        res["E_M"] = macro_fn(res["E_m"], n_boids)
        # plt.scatter(res["C_M"], res["E_M"], alpha=.2)
        # plt.show()
        if not compute_psi:
            continue
        psi, I_macro, I_up = psi_score(
            res["C_m"], res["C_M"], res["E_M"], n_boids,
            k=K, n_surrogates=N_SURROGATES,
        )
        res["psi"] = psi
        res["sum_up"] = float(I_up.sum())
        res["I_macro"] = I_macro
    show_line(regime_name, macro_name, prior, results)
    return results

def main():
    print(f"Sweep: N={N_SAMPLES}, lag={LAG}, n_surrogates={N_SURROGATES}, k={K}")
    print(f"       n_boids={HP_BASE['n_boids']}, n_steps={HP_BASE['n_steps']}, n_repeat={N_REPEAT}")
    print()

    header = (
        f"{'regime':<9} {'macro':<13} {'prior':<15} "
        f"{'mean_C':>7} {'std_C':>7} {'saturated':>10} | {'I_macro':>12} {'sum_up':>12} {'Psi':>12}"
    )
    print(header)
    print("-" * len(header))

    for regime_name in REGIMES:
        for prior in PRIORS:
            results = get_results_micro(prior, regime_name)
            for macro_name in MACROS:
                results = get_results_macro(prior, regime_name, macro_name, results)
        print()

def show_intervention():
    run_id = 0
    step_id = 50
    hp = HP_BASE | HP_ATTRACTOR_DEFAULT
    traj, Z_traj, A_traj = sample_trajectory_confounder(hp, N_SAMPLES_REF, init_lag=INIT_LAG, verbose=1, path="data/attractor.npy")
    # traj_flat = traj.reshape(np.prod(traj.shape[:2]), np.prod(traj.shape[2:]))
    C_m_ref = traj[run_id,step_id]
    C_M_ref = MACROS["distance"](C_m_ref.reshape(1,1,-1), hp["n_boids"])
    print("original initial V:", C_M_ref)
    C_m_spread = make_interventions_spread(C_m_ref[None, None, ...].copy(), hp, N=1)[0]
    C_M_spread = MACROS["distance"](C_m_spread.reshape(1,1,-1), hp["n_boids"])
    print("spread initial V:", C_M_spread)

    E_m_ref = traj[run_id,step_id+LAG]
    E_M_ref = MACROS["distance"](E_m_ref.reshape(1,1,-1), hp["n_boids"])
    print("original final V:", E_M_ref)

    traj_spread, _, _ = make_sweep_confounder(
        C_m_ref[None, ...], Z_traj[run_id,step_id].reshape(1), 
        A_traj[run_id,step_id].reshape(1,2), {**hp, "n_steps": LAG}, verbose=False
        )

    E_m_spread = traj_spread[0,-1]
    E_M_spread = MACROS["distance"](E_m_spread.reshape(1,1,-1), hp["n_boids"])
    print("spread final V:", E_M_spread)


    from render_simulation import plot_simulation, plot_boids
    _, axes = plt.subplots(2,2,figsize=(9,4))
    plot_boids(C_m_ref, axes[0,0])
    plot_boids(C_m_spread, axes[0,1])
    plot_boids(E_m_ref, axes[1,0])
    plot_boids(E_m_spread, axes[1,1])
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
    
    # make_figure("tight")
    # make_noise_sweep("tight", "distance", 15, .7, 16)
    # render_noise_sweep("tight", "distance", 15, .7, 5, show_legend=True, show_bar=False, axes_w=2.2)
    # make_noise_sweep("moderate", "distance", 15, .7, 16)
    # render_noise_sweep("moderate", "distance", 15, .7, 5, False, False)
    # make_noise_sweep("attractor", "distance", 15, .7, 16)
    # render_noise_sweep("attractor", "distance", 15, .7, 5, False, False, axes_w=2)
    # make_noise_sweep("attractor-loose", "distance", 15, .7, 16)
    # render_noise_sweep("attractor-loose", "distance", 15, .7, 5, show_bar=True, show_legend=False, axes_w=2.2)