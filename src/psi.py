import numpy as np
from tqdm import tqdm

from mi_estimators import (
    ksg_mi,
    mi_scalar_bias_corrected,
)
from flocking_observables import observe_polarization
from basic_example import MI_from_data


def psi_score(C_m, C_M, E_M, n_boids, k=5, n_surrogates=100, verbose=False):
    """
    Generic Rosas Psi for any pre-computed scalar or vector macro.

        Psi = I(C_M, E_M) - sum_{j=1..n_boids} I(C_m^j, E_M)

    Parameters
    ----------
    C_m : array, shape (n_samples, n_boids * 3)
        Raw micro states at time t.
    C_M : array, shape (n_samples,) or (n_samples, d)
        Macro feature at time t.
    E_M : array, shape (n_samples,) or (n_samples, d)
        Macro feature at time t+1.
    n_boids : int
    k : int, default=5
    n_surrogates : int, default=100

    Returns
    -------
    psi : float
    I_macro : float
    I_upward_sum : float
    """
    C_m = np.asarray(C_m, dtype=float)
    C_M = np.asarray(C_M, dtype=float)
    E_M = np.asarray(E_M, dtype=float)
    assert C_m.shape[1] == n_boids * 3, \
        f"C_m must have shape (n_samples, n_boids*3); got {C_m.shape}"
    assert C_M.shape[0] == E_M.shape[0] == C_m.shape[0], \
        "C_M, E_M, C_m must share the sample axis"

    I_macro = mi_scalar_bias_corrected(
        C_M, E_M,
        n_permutations=n_surrogates, estimator=ksg_mi, k=k, verbose=verbose,
    )
    if verbose:
        print(f"I_macro = {I_macro:.4f}")

    I_upward = []
    for j in range(n_boids):
        C_j = C_m[:, j * 3: (j + 1) * 3]
        I_j = mi_scalar_bias_corrected(
            C_j, E_M,
            n_permutations=n_surrogates, estimator=ksg_mi, k=k, verbose=verbose,
        )
        I_upward.append(I_j)
        if verbose:
            print(f"  boid {j}: I = {I_j:.4f}")
    I_upward = np.asarray(I_upward)

    psi = float(I_macro - I_upward.sum())
    if verbose:
        print(f"I_upward sum = {I_upward.sum():.4f}")
        print(f"Psi = {psi:.4f}")
    return psi, float(I_macro), I_upward