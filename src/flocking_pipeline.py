"""
Data-generation helpers around the flocking simulation.

These wrap `flocking.init_boids_batch` and `flocking.make_sweep` to produce
trajectory and pair samples in shapes that the MI / Psi machinery expects.

Functions
---------
- sample_trajectory_boids : run n_runs simulations and return a
  (n_runs, T_eff, n_boids, 3) trajectory tensor (frames after init_lag).
- reshape_trajectory      : turn that tensor into pair samples of shape
  (N, 2, n_boids*3) at a given `lag` (C_t, C_{t+lag}).
- build_pairs             : high-level helper that calls the two functions above
  and also handles the "interventions" prior via sample_pairs_boids.
- sample_pairs_boids      : alternative pair sampler that runs `lag` steps from
  draws sampled under one of two priors:
      "uniform"        — uniform over the bounding box of the trajectory cloud
      "interventions"  — resample from trajectory cloud, then perturb headings
- make_interventions_replace / make_interventions_noise : the two perturbation
  flavours used by `sample_pairs_boids` under prior="interventions".
- make_initial_conditions_uniform_boids :
  uniform draws from the bounding box of trajectory frames.

The `pairs` helper from `basic_example` is re-exported here for convenience.
"""

import numpy as np
import os

from flocking import WIDTH, init_boids_batch, make_sweep, pi
from basic_example import pairs  
from confounder_simulations import init_confounder_state, make_sweep_confounder, HP_ATTRACTOR_DEFAULT


def make_initial_conditions_uniform_boids(runs, N):
    """N uniform draws from the bounding box of the t=0 frames of `runs`."""
    return np.random.uniform(
        runs[:, 0].min(0),
        runs[:, 0].max(0),
        size=(N, runs.shape[-1]),
    )

def sample_trajectory_boids(hp, N, init_lag, verbose=1, path=None):
    """
    Run n_runs simulations and keep frames after `init_lag`, every `lag`-th step.

    Picks n_runs so that ~N pair samples result after `reshape_trajectory`.
    """
    if path is not None and os.path.isfile(path):
        return np.load(path)
    N_samples_per_run = hp["n_steps"] - init_lag
    n_runs = N // N_samples_per_run
    states = init_boids_batch(hp, n_runs, seed=hp["seed"])
    trajectories = make_sweep(states, hp, verbose)
    trajectories = trajectories[:, init_lag:]
    if path is not None:
        np.save(path, trajectories)
    return trajectories

def sample_trajectory_confounder(hp, N, init_lag=20, verbose=1, path=None):
    """
    Run n_runs confounder simulations; return boid trajectories after init_lag.

    This is a drop-in replacement for
    ``flocking_pipeline.sample_trajectory_boids``.
    Returns the same (n_runs, T_eff, n_boids, 3) tensor, so
    ``reshape_trajectory`` and all downstream Psi machinery work unchanged.

    The confounder timeseries Z_t is discarded (it is hidden from the
    experimenter), keeping the micro state space identical to the real
    flocking scenario.

    Parameters
    ----------
    hp       : dict  must include 'confounder_type' and all relevant keys
    N        : int   target number of (C, E) sample pairs after reshape
    init_lag : int   frames to discard at the start of each run
    verbose  : int
    path     : str or None  if given and file exists, load and return cached array

    Returns
    -------
    trajectories : (n_runs, T_eff, n_boids, 3)
    """
    z_path = path.split(".")[0] + "_z.npy"
    a_path = path.split(".")[0] + "_a.npy"
    if path is not None and os.path.isfile(path):
        traj = np.load(path)
        Z = np.load(z_path)
        if hp["confounder_type"] == "attractor":
            A = np.load(a_path)
        else:
            A = None
        return traj, Z, A

    N_samples_per_run = hp["n_steps"] - init_lag  # T_eff = n_steps - init_lag + 1 frames, so n_steps-init_lag pairs
    n_runs = max(1, N // N_samples_per_run)

    boids, Z0, A0 = init_confounder_state(hp, n_runs, seed=hp.get("seed"))
    histories, Z_histories, A_histories = make_sweep_confounder(boids, Z0, A0, hp, verbose=bool(verbose))

    # Discard burn-in frames; confounder is already near stationary
    trajectories = histories[:, init_lag:]
    z_trajectories = Z_histories[:, init_lag:]
    if A_histories is not None:
        a_trajectories = A_histories[:, init_lag:]
    else:
        a_trajectories = None

    if path is not None:
        np.save(path, trajectories)
        np.save(z_path, z_trajectories)
        if hp["confounder_type"] == "attractor":
            np.save(a_path, a_trajectories)


    return trajectories, z_trajectories, a_trajectories

def reshape_trajectory(trajectories, lag):
    """
    Turn (n_runs, T, n_boids, 3) into pair samples (N, 2, n_boids*3),
    where N = n_runs * (T - 1) and the pair axis indexes (t, t+1).
    """
    C = trajectories[:, :-lag]
    E = trajectories[:, lag:]
    a, b, c, d = C.shape
    C = C.reshape(a * b, c * d)
    E = E.reshape(a * b, c * d)
    X = np.array([C, E]).transpose(1, 0, 2)
    return X

def make_interventions_noise(init_boids, hp, N, n_interventions, noise_factor):
    """
    Perturb n_interventions randomly chosen micro variables per sample by
    Gaussian noise scaled to noise_factor * observed_range of that variable.

    This keeps perturbed states within the trajectory cloud (unlike the uniform
    draw) while breaking short-range statistical dependencies. The known
    limitation is that position ranges span the full arena (~500 units), so
    even a small noise_factor ejects boids from their neighbors' view radius
    (~25 units) before the confounder's global restoring force is overcome —
    the root cause of the discrimination failure reported in Section IV-D.
    """
    if n_interventions == 0 or abs(noise_factor) < .01:
        return init_boids

    init_boids = init_boids.reshape(N, hp["n_boids"] * 3)
    n, d = init_boids.shape

    feature_ranges = np.ptp(init_boids, axis=0)  # shape: (d,)

    e_features = np.random.choice(d, size=(n, n_interventions))
    e_values = np.take_along_axis(init_boids, e_features, axis=1)
    scales = feature_ranges[e_features]
    e_values = e_values + np.random.normal(loc=0, scale=noise_factor * scales)

    np.put_along_axis(init_boids, e_features, e_values, axis=1)
    init_boids = init_boids.reshape(N, hp["n_boids"], 3)
    init_boids[...,:2] = (init_boids[...,:2] + WIDTH) % WIDTH
    init_boids[...,2] = (init_boids[...,2] + 2 * pi) % (2*pi)
    return init_boids

def confounder_init_interventions_uniform(sample, shape):
    if sample is not None:
        return np.random.uniform(sample.min(),sample.max(),size=shape)

def confounder_init_interventions_noise(
        sample, ids, intervention_factor=.5, noise_factor=.5, low=0, up=1,
        ):
    if sample is not None:
        init = sample[ids]
        e_ids = np.random.rand(*init.shape) < intervention_factor
        init += np.random.normal(loc=0, scale=noise_factor, size=init.shape) * e_ids
        return np.clip(init, low, up)

def reshape_confounder(Z_traj):
    if Z_traj is None:
        return
    Z_C = Z_traj[:, 0]
    Z_E = Z_traj[:,-1]
    return np.array([Z_C, Z_E]).T

def reshape_traj_pairs(trajectories):
    """
    Extract the first and last frame of each rollout to form (C_t, C_{t+lag})
    pairs of shape (N, 2, n_boids*3).

    Used by sample_pairs_boids after running lag steps from a resampled or
    perturbed initial condition, so that the lag is baked into the dynamics
    rather than read off a pre-recorded trajectory.
    """
    C = trajectories[:, :1]
    E = trajectories[:, -1:]
    a, b, c, d = C.shape
    C = C.reshape(a * b, c * d)
    E = E.reshape(a * b, c * d)
    return np.array([C, E]).transpose(1, 0, 2)

def sample_pairs_boids(samples, hp, N, lag, prior, 
                       verbose=1, n_interventions=10, noise_factor=.25, 
                       Z_sample=None, Z_intervention_factor=.5,
                       A_sample=None, A_intervention_factor=.25,
                       ):
    """
    Generate N pair samples (C_t, C_{t+lag}) under the chosen prior over C_t.
    Inputs are flattened: samples (N_samples, N_features) and Z_sample (N_samples,)

    prior:
        "uniform"        — uniform over the bounding box of `runs`.
        "traj"           — resample with replacement from `runs`.
        "interventions"  — resample, then perturb headings via make_interventions_noise.

    Returns X of shape (N, 2, n_boids*3).
    """
    np.random.seed(hp["seed"])
    # Uniform intervention distirbution
    ids = np.random.choice(range(samples.shape[0]), size=N)
    if prior == "uniform":
        init_boids = np.random.uniform(
            samples.min(0),
            samples.max(0),
            size=(N, samples.shape[-1]),
        )
        init_boids = init_boids.reshape(N, hp["n_boids"], 3)
        Z_init = confounder_init_interventions_uniform(Z_sample, N)
        A_init = confounder_init_interventions_uniform(A_sample, (N,2))
    
    # Perturbation based intervention distribution
    elif prior == "noise":
        init_boids = samples[ids].reshape(N, hp["n_boids"], 3)
        init_boids = make_interventions_noise(init_boids, hp, N, n_interventions, noise_factor)
        # Z_init = confounder_init_interventions_noise(Z_sample, ids, Z_intervention_factor, noise_factor, 0, 1)
        # A_init = confounder_init_interventions_noise(A_sample, ids, A_intervention_factor, noise_factor, 0, WIDTH)
        Z_init = Z_sample[ids].reshape(-1) if Z_sample is not None else None
        A_init = A_sample[ids].reshape(-1, 2) if A_sample is not None else None
    elif prior == "shift":
        init_boids = samples[ids].reshape(N, hp["n_boids"], 3)
        init_boids = make_interventions_spread(init_boids, hp, N)#, macro_name)#, n_interventions, noise_factor)
        if Z_sample is not None:
            Z_init = Z_sample[ids].reshape(-1)
        if A_sample is not None:
            A_init = A_sample[ids].reshape(-1, 2)
        # Z_init = confounder_init_interventions_noise(Z_sample, ids, Z_intervention_factor, noise_factor, 0, 1)
        # A_init = confounder_init_interventions_noise(A_sample, ids, A_intervention_factor, noise_factor, 0, WIDTH)      
    elif prior == "confounder":
        init_boids = samples[ids].reshape(N, hp["n_boids"], 3)
        if Z_sample is not None:
            Z_init = np.random.uniform(0,1,size=len(ids))
        else:
            Z_init = None
        if A_sample is not None:
            
            A_init = A_sample[ids].reshape(-1, 2)
        else:
            A_init = None
    else:
        raise Exception(f"Prior {prior} not recognized")
    
    # Observation distribution
    if hp["confounder"]:
        trajectories, Z_traj, A_traj = make_sweep_confounder(init_boids, Z_init, A_init, {**hp, "n_steps": lag}, verbose=False)
        Z = reshape_confounder(Z_traj)
        A = reshape_confounder(A_traj)
    else:
        trajectories = make_sweep(init_boids, {**hp, "n_steps": lag}, verbose)
        Z = None
        A = None
    X = reshape_traj_pairs(trajectories)
    return X, Z, A

def build_pairs(hp, prior, lag, init_lag, n_samples_ref, n_samples, n_interventions, noise_factor, path=None):
    """Return X of shape (N, 2, n_boids*3)."""
    if hp["confounder"]:
        traj, Z_traj, A_traj = sample_trajectory_confounder(hp, n_samples_ref, init_lag=init_lag, verbose=1, path=path)
    else:
        traj = sample_trajectory_boids(hp, N=n_samples_ref, init_lag=init_lag, verbose=0, path=path)
        Z_traj = None
        A_traj = None
    if prior == "traj":
        shaped_traj = reshape_trajectory(traj, lag)
        sample_ids = np.random.choice(range(shaped_traj.shape[0]), size=n_samples, replace=False)
        return shaped_traj[sample_ids], Z_traj, A_traj
    
    traj_flat = traj.reshape(np.prod(traj.shape[:2]), np.prod(traj.shape[2:])).copy()
    if Z_traj is not None:
        Z_traj = Z_traj.reshape(-1)
    if A_traj is not None:
        A_traj = A_traj.reshape(-1, 2)
    return sample_pairs_boids(traj_flat, hp, N=n_samples,
                              n_interventions=n_interventions, noise_factor=noise_factor,
                              lag=lag, prior=prior, verbose=0, Z_sample=Z_traj, A_sample=A_traj)