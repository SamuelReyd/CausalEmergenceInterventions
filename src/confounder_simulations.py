import numpy as np
import os
from tqdm import tqdm

from flocking import (
    pi, WIDTH,
    direction,
    init_boids_batch,
    get_filters,
    turn_towards,
)


# ---------------------------------------------------------------------------
# Default hp dicts
# ---------------------------------------------------------------------------

HP_WIND_DEFAULT = dict(
    # Standard boid parameters (positions/speeds only — interaction rules unused)
    # Identifies this as a confounder sim (disables real interactions in make_sweep)
    confounder=True,       
    confounder_type="wind",
    # AR(1) process for Z_t ∈ [0, 1]
    z_ar1_rho=0.97,
    z_ar1_noise=0.20,
    wind_turn_speed=5 * pi / 180,   # max turn per step toward ψ_t (radians)
    wind_psi_drift=2 * pi / 180,    # std of ψ_t random walk per step (radians)
    wind_noise=1 * pi / 180,                # isotropic noise on boid headings each step (radians)
)

HP_ATTRACTOR_DEFAULT = dict(
    confounder=True,
    confounder_type="attractor",
    # AR(1) process for Z_t ∈ [-1, 1]
    z_ar1_rho=0.97,
    z_ar1_noise=0.20,
    # Attractor-specific
    attractor_alpha=0.15,     
    attractor_noise=1.0,      
    radius_min=5,
    radius_max=40
)


# ---------------------------------------------------------------------------
# AR(1) confounder process
# ---------------------------------------------------------------------------

def _ar1_step(Z, rho, sigma, low, high, mean=0.0):
    """
    One step of a mean-reverting clipped AR(1) process:

        Z_{t+1} = clip(mean + rho*(Z_t - mean) + noise_scale * eps, low, high)

    The noise is scaled by sqrt(1 - rho^2) * sigma so the stationary std of
    the *unclipped* process is approximately `sigma`.

    Parameters
    ----------
    Z     : (B,) current values
    rho   : float, autocorrelation (0 < rho < 1)
    sigma : float, target stationary std of the unclipped process
    low, high : float, hard clip bounds
    mean  : float, mean-reversion target (default 0.0)

    Returns
    -------
    Z_new : (B,)
    """
    eps = np.random.standard_normal(Z.shape)
    noise_scale = np.sqrt(max(1.0 - rho ** 2, 0.0)) * sigma
    Z_new = mean + rho * (Z - mean) + noise_scale * eps
    return np.clip(Z_new, low, high)


def _init_Z(B, low, high, rho, sigma, mean=0.0, n_burnin=500, seed=None):
    """
    Initialise B independent AR(1) chains near their stationary distribution
    by running `n_burnin` steps from the mean-reversion target.

    Returns Z of shape (B,).
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.full(B, mean, dtype=float)
    for _ in range(n_burnin):
        Z = _ar1_step(Z, rho, sigma, low, high, mean=mean)
    return Z


# ---------------------------------------------------------------------------
# Wind-coherence step
# ---------------------------------------------------------------------------

def _update_wind(s, Z, psi, hp):
    """
    One step of the wind-coherence confounder simulation.

    Each boid independently turns toward the shared reference direction ψ
    with a max_turn proportional to Z_t. No boid observes any other boid.

    Parameters
    ----------
    s   : (B, N, 3)  current boid states
    Z   : (B,)       current coherence level ∈ [0, 1]
    psi : (B,)       current reference direction (radians)
    hp  : dict

    Returns
    -------
    s_new : (B, N, 3)
    Z_new : (B,)
    psi_new : (B,)
    """
    B, N, _ = s.shape
    rho   = hp["z_ar1_rho"]
    sigma = hp["z_ar1_noise"]

    # --- Update confounder Z_t+1 ∈ [0, 1], mean-reverting to 0.5 ---
    Z_new = _ar1_step(Z, rho, sigma, low=0.0, high=1.0, mean=0.8)

    # --- Update shared reference direction ψ_t+1 ---
    psi_drift = np.random.normal(0, hp["wind_psi_drift"], size=B)
    psi_new = (psi + psi_drift) % (2 * pi)

    # --- Turn each boid independently toward ψ ---
    # max_turn scales with Z: when Z=0 boids ignore ψ entirely (→ drift freely)
    # when Z=1 boids snap toward ψ at full wind_turn_speed.
    headings = s[:, :, 2]                        # (B, N)

    # Broadcast ψ to (B, N) — all boids in a run share the same target
    target = psi[:, None] * np.ones((B, N))      # (B, N)

    # Effective max_turn per boid: Z scales the turn speed
    effective_max_turn = Z[:, None] * hp["wind_turn_speed"]  # (B, N)

    # When Z ≈ 0 the target is irrelevant; we still need to add free diffusion
    # so headings don't freeze. Use wind_psi_drift as free diffusion noise.
    free_noise_std = (1.0 - Z[:, None]) * hp["wind_psi_drift"] * 10
    # Build composite turn: toward ψ (strength Z) + isotropic diffusion (strength 1-Z)
    delta_to_target = (target - headings + pi) % (2 * pi) - pi   # shortest arc
    clamped = np.sign(delta_to_target) * np.minimum(
        np.abs(delta_to_target), effective_max_turn
    )
    diffusion = np.random.normal(0, free_noise_std + 1e-8, size=(B, N))
    noise = np.random.normal(0, hp["wind_noise"], size=(B, N))
    new_headings = (headings + clamped + diffusion + noise) % (2 * pi)

    # --- Update positions (straight-line motion, toric arena) ---
    new_pos = s[:, :, :2] + direction(new_headings) * hp["max_speed"]
    new_pos = new_pos % WIDTH

    s_new = np.concatenate([new_pos, new_headings[:, :, None]], axis=-1)
    return s_new, Z_new, psi_new


# ---------------------------------------------------------------------------
# Signed-attractor step
# ---------------------------------------------------------------------------

def _update_attractor(s, Z, A, hp):
    B, N, _ = s.shape
    rho, sigma = hp["z_ar1_rho"], hp["z_ar1_noise"]
    Z_new = _ar1_step(Z, rho, sigma, low=0, high=1, mean=.5)

    # --- Z -> target group radius ---
    R_min, R_max = hp["radius_min"], hp["radius_max"]
    R_target = R_min + (R_max - R_min) * Z          # (B,)

    pos = s[:, :, :2]
    radial = pos - A[:, None, :]
    dist = np.linalg.norm(radial, axis=-1, keepdims=True)         # (B,N,1)
    safe_dist = np.where(dist > 1e-6, dist, 1.0)
    unit_radial = radial / safe_dist

    # boids at the exact center have no outward direction -> give them one,
    # otherwise they can never be pushed out to R(Z)
    rand_ang = np.random.uniform(0, 2 * pi, size=dist.shape[:-1])
    rand_dir = np.stack([np.cos(rand_ang), np.sin(rand_ang)], axis=-1)
    unit_radial = np.where(dist > 1e-6, unit_radial, rand_dir)

    # --- spring toward rest length R(Z) ---
    alpha = hp["attractor_alpha"]
    radial_error = R_target[:, None, None] - dist                 # >0 push out, <0 pull in
    displacement = alpha * radial_error * unit_radial             # (B,N,2)

    noise = np.random.normal(0, hp["attractor_noise"], size=(B, N, 2))
    new_pos = pos + displacement + noise

    net_move = displacement + noise
    net_norm = np.linalg.norm(net_move, axis=-1)
    has_move = net_norm > 1e-8
    new_headings = np.where(
        has_move,
        (np.arctan2(net_move[..., 1], net_move[..., 0]) + 2 * pi) % (2 * pi),
        s[:, :, 2],
    )

    s_new = np.concatenate([new_pos, new_headings[:, :, None]], axis=-1)
    s_new[:, :, :2] = s_new[:, :, :2] % WIDTH
    return s_new, Z_new, A


# ---------------------------------------------------------------------------
# Unified make_sweep_confounder
# ---------------------------------------------------------------------------

def make_sweep_confounder(boids, Z0, A0, hp, verbose=False):
    """
    Run B confounder simulations in parallel.

    Parameters
    ----------
    boids   : (B, N, 3)  initial boid states
    Z0      : (B,)       initial confounder values
    A0      : (B,2)
    hp      : dict       must include 'confounder_type' and related keys

    Returns
    -------
    histories   : (B, n_steps+1, N, 3)   boid state histories
    Z_histories : (B, n_steps+1)          confounder value histories
    A_histories : (B, n_steps+1, 2)
    """
    if boids.ndim == 2:
        boids = boids[None, ...]
    # print(boids.shape, Z0.shape, )
    assert len(boids.shape) == 3 and len(Z0.shape) == 1 and (A0 is None or len(A0.shape) == 2)
    s = boids.copy()
    B, N, _ = s.shape
    n_steps = hp["n_steps"]
    ctype = hp["confounder_type"]

    histories   = np.empty((B, n_steps + 1, N, 3), dtype=np.float64)
    Z_histories = np.empty((B, n_steps + 1),        dtype=np.float64)
    if A0 is not None:
        A_histories = np.empty((B, n_steps + 1, 2),        dtype=np.float64)
    else:
        A_histories = None
    histories[:, 0]   = s
    Z_histories[:, 0] = Z0
    if A0 is not None:
        A_histories[:, 0] = A0

    Z = Z0.copy()
    if A0 is not None:
        A = A0.copy()

    if ctype == "wind":
        # Initialise shared reference direction per run: mean of initial headings
        psi = s[:, :, 2].mean(axis=1)   # (B,) — start aligned with initial flock
        for t in tqdm(range(n_steps), disable=not verbose,
                      desc="Simulating wind confounder"):
            s, Z, psi = _update_wind(s, Z, psi, hp)
            histories[:, t + 1]   = s
            Z_histories[:, t + 1] = Z

    elif ctype == "attractor":
        for t in tqdm(range(n_steps), disable=not verbose,
                      desc="Simulating attractor confounder"):
            s, Z, A = _update_attractor(s, Z, A, hp)
            histories[:, t + 1]   = s
            Z_histories[:, t + 1] = Z
            A_histories[:, t + 1] = A

    else:
        raise ValueError(
            f"Unknown confounder_type {ctype!r}. Must be 'wind' or 'attractor'."
        )

    return histories, Z_histories, A_histories


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_boids_uniform(hp, B, seed=None):
    if seed is not None:
        np.random.seed(seed)
    boids = np.zeros((B,hp["n_boids"],3), dtype=float)
    boids[:,:,:2] = np.random.uniform(0, WIDTH, size=(B,hp["n_boids"],2))
    boids[:,:,2]  = np.random.uniform(0, 2*pi, size=(B,hp["n_boids"]))
    return boids

def init_confounder_state(hp, B, seed=None):
    """
    Initialise B runs: boid positions (via init_boids_batch) and Z_0.

    For 'wind':      Z_0 ∈ [0, 1], initialised near stationary distribution.
    For 'attractor': Z_0 ∈ [-1, 1], mean-zero stationary distribution.

    Returns
    -------
    boids : (B, N, 3)
    Z0    : (B,)
    """
    
    ctype = hp["confounder_type"]
    rho   = hp["z_ar1_rho"]
    sigma = hp["z_ar1_noise"]

    if ctype == "wind":
        boids = init_boids_batch(hp, B, seed=seed)
        Z0 = _init_Z(B, low=0.0, high=1.0, rho=rho, sigma=sigma, mean=0.8, seed=seed)
        A0 = None
    elif ctype == "attractor":
        boids = init_boids_uniform(hp, B, seed=None)
        Z0 = _init_Z(B, low=0., high=1.0, rho=rho, sigma=sigma, mean=0.5, seed=seed)
        A0 = np. random.randint(
            HP_ATTRACTOR_DEFAULT["radius_max"], 
            WIDTH-HP_ATTRACTOR_DEFAULT["radius_max"], 
            size=(B,2))

    else:
        raise ValueError(f"Unknown confounder_type {ctype!r}.")

    return boids, Z0, A0

