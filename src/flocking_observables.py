import numpy as np

def reshaper_decorator(function):
    """
    Wrap an observable that expects shape (-1, n_boids, 3) so it accepts any of:
    (N, n_boids*3), (N, n_boids, 3), (N, 2, n_boids*3), (N, 2, n_boids, 3).

    The output preserves all leading axes and gets a trailing feature axis (-1).
    """
    def new_func(samples, n_boids, *args, **kwargs):
        shape = samples.shape
        X = samples.reshape(-1, n_boids, 3)
        ret = function(X, n_boids, *args, **kwargs)
        ret = ret.reshape(*shape[:-1], -1)
        return ret
    return new_func

def observe_dim(X, dim, n_boids, mean=True):
    """
    Mean or median of one component (x=0, y=1, theta=2) across boids.
    Accepts any leading shape; the trailing flat axis must equal n_boids*3.
    """
    shape = X.shape
    X = X.reshape(-1, n_boids, 3)
    if mean:
        V = X[..., dim].mean(axis=-1)
    else:
        V = np.median(X[..., dim], axis=-1)
    return V.reshape(shape[:-1])

@reshaper_decorator
def observe_radius(X, n_boids, mean=False):
    """Median (or mean) squared distance of each boid to the flock center of mass."""
    xy = X[..., :2]
    center = np.mean(xy, axis=1, keepdims=True)
    if mean:
        radius = np.mean(np.sum((xy - center) ** 2, axis=2), axis=1)
    else:
        radius = np.median(np.sum((xy - center) ** 2, axis=2), axis=1)
    return radius

def observe_polarization(X, n_boids):
    """
    Vicsek polarization (order parameter) of a flock.

        phi = (1/N) * | sum_j exp(1j * theta_j) |     in [0, 1]

    1 = perfectly aligned, 0 = uniformly random headings.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_boids * 3)
        Micro states with (x, y, theta) per boid, flattened.
    n_boids : int

    Returns
    -------
    phi : ndarray, shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    headings = X.reshape(X.shape[0], n_boids, 3)[:, :, 2]
    return np.abs(np.mean(np.exp(1j * headings), axis=1))

def _eff_pairwise_dist(pos):
    """
    Squared pairwise distances for (B, N, 2) positions: ‖x_i - x_j‖^2.

    Implemented via ‖x_i‖^2 + ‖x_j‖^2 - 2 x_i·x_j; numerical noise that
    would push the diagonal slightly negative is clipped to 0.
    """
    sq_norms = (pos ** 2).sum(axis=-1)             # (B, N)
    sq_dist = (
        sq_norms[:, :, None]
        + sq_norms[:, None, :]
        - 2 * (pos @ pos.swapaxes(-1, -2))
    )                                              # (B, N, N)
    np.clip(sq_dist, 0, None, out=sq_dist)
    return sq_dist

@reshaper_decorator
def observe_pairwise(X, n_boids, mean=False):
    """Median (or mean) squared pairwise distance between boids."""
    pos = X[..., :2]
    sq_dist = _eff_pairwise_dist(pos)
    if mean:
        dists = sq_dist.mean(axis=1).mean(axis=1)
    else:
        dists = np.median(np.median(sq_dist, axis=1), axis=1)
    return dists

def dist_factor(dists, d0, d1):
    """
    Piecewise-linear soft threshold:
        d <= d0 -> 1
        d >= d1 -> 0
        otherwise linear interp from 1 at d0 to 0 at d1.
    """
    dists = dists.copy()
    f0 = dists < d0
    f1 = dists > d1
    dists[f0] = 1
    dists[f1] = 0
    dists[~(f1 | f0)] = 1 - (dists[~(f1 | f0)] - d0) / (d1 - d0)
    return dists

def log_pol(values, do_log=True, eps=None):
    """Optional -log(1 - phi) transform of polarization values, with floor eps."""
    if do_log:
        values = 1 - values
        if eps is None:
            eps = np.min(values[values > 0])
        return -np.log(np.clip(values, eps, 1))
    return values

def saturate(values, low, high):
    """
    Piecewise-linear saturation onto [0, 1]:
        v <= low  -> 0
        v >= high -> 1
        otherwise linear from 0 at `low` to 1 at `high`.

    The complement of `dist_factor` (which maps low->1, high->0).
    """
    v = np.asarray(values, dtype=float)
    return np.clip((v - low) / (high - low), 0.0, 1.0)

def observe_polarization_sat(X, n_boids, low=0.5, high=1.0):
    """
    Polarization with low-end saturation.

    Random headings on N=10 boids give phi ~ 1/sqrt(N) ~ 0.32. With
    low=0.5, anything that looks "less coherent than random clusters of
    half-the-flock-aligned" maps to 0.
    """
    return saturate(observe_polarization(X, n_boids), low, high)

def observe_pairwise_factor(X, n_boids, min_sep, fact0=0.0, fact1=10.0):
    """
    Median squared pairwise distance pushed through `dist_factor`.

        d <= (min_sep * sqrt(fact0))^2  -> 1   (tightly clustered)
        d >= (min_sep * sqrt(fact1))^2  -> 0   (dispersed)

    Returns shape (n_samples,). Wraps `coarse_grain` with discrete=False
    but strips the trailing axis added by `observe_pairwise`'s decorator.
    """
    pw = observe_pairwise(X, n_boids, mean=False)
    # print(pw)
    # pw = observe_radius(X, n_boids, mean=False)
    factor = dist_factor(pw, min_sep ** 2 * fact0, min_sep ** 2 * fact1)
    return factor.ravel()

def observe_polarization_x_pairwise(X, n_boids, min_sep,
                                     pol_low=0.5, pol_high=1.0,
                                     dist_fact0=0.0, dist_fact1=10.0):
    """
    Product of saturated polarization and pairwise dist_factor.
    Only > 0 when the swarm is both heading-aligned and spatially clustered.
    """
    p = observe_polarization_sat(X, n_boids, pol_low, pol_high)
    d = observe_pairwise_factor(X, n_boids, min_sep, dist_fact0, dist_fact1)
    return p * d

def observe_com_displacement(X, n_boids):
    """
    Center-of-mass position, centered on the sample-wise mean CoM.

    Returns (n_samples, 2). Subtracting the dataset mean removes any
    absolute-frame bias so that "uniform initial conditions" produces a
    near-zero-mean cloud of CoM positions.
    """
    X = np.asarray(X, dtype=float)
    X = X.reshape(X.shape[0], n_boids, 3)
    com = X[:, :, :2].mean(axis=1)            # (n_samples, 2)
    return com - com.mean(axis=0, keepdims=True)

def observe_com_displacement_sat(X, n_boids, low=5.0, high=25.0):
    """
    Scalar CoM-displacement magnitude with saturation.

    Small displacements (CoM near the sample mean — typical of the uniform
    prior, where N=10 boids drawn in a 50-unit box give CoM std ~ 4.6) map to
    0; large displacements (flocking CoM that has migrated) saturate at 1.
    """
    com = observe_com_displacement(X, n_boids)
    mag = np.linalg.norm(com, axis=1)
    return saturate(mag, low, high)

# def coarse_grain(X, n_boids, min_sep, fact0, fact1):
#     """
#     Macro feature pipeline currently used in the buggy Psi: a distance-based
#     coarse-graining of pairwise distances.

#     Note: do_log is accepted for backwards compatibility with earlier code paths
#     that combined polarization (see commented lines); it is not currently used
#     because the polarization branch is disabled.
#     """
#     V = dist_factor(
#         observe_pairwise(X, n_boids, mean=False),
#         min_sep ** 2 * fact0,
#         min_sep ** 2 * fact1,
#     )
#     return V

def obs(X, hp, dim):
    """
    Legacy dispatcher kept for reference.

    Note: this calls `observe_polarization(X, hp)` and `observe_dim(X, dim, hp)`
    passing `hp` (a dict) as the `n_boids` argument. It would error if invoked
    today. Preserved here so the move from the notebook is faithful.
    """
    if dim < 0:
        return observe_polarization(X, hp)
    else:
        return observe_dim(X, dim, hp)
