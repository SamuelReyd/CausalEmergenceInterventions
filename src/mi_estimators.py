"""
Mutual-information estimators and binning helpers.

Provides:
- ksg_mi, ksg_mi_bias_corrected : KSG MI estimator (continuous, any dim) and
  a permutation-based bias-correction wrapper.
- ksg_mi_scalar : backwards-compatible alias of ksg_mi (it now handles 1D
  inputs natively).
- kde_mi_scalar : KDE plug-in MI estimator for scalar pairs, with Silverman /
  CV bandwidth helpers.
- mi_scalar_bias_corrected : permutation-based bias correction around any
  estimator (defaults to ksg_mi_scalar).
- discretize, discretize_independent, discretize_joint : uniform-binning
  helpers with shape conventions tailored to (samples, [pair], features).
"""

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import KDTree, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


# ---------------------------------------------------------------------------
# KSG estimator
# ---------------------------------------------------------------------------

def ksg_mi(C, E, k=5, standardize=True, leaf_size=40):
    """
    KSG mutual information estimator for continuous variables.

    Accepts 1D or 2D inputs:
        - 1D arrays of shape (n,) are treated as (n, 1).
        - 2D arrays of shape (n, d) are used as-is.

    Parameters
    ----------
    C : array-like, shape (n,) or (n, p)
    E : array-like, shape (n,) or (n, q)
    k : int, default=5
    standardize : bool, default=True
    leaf_size : int, default=40

    Returns
    -------
    mi : float
        Estimated mutual information in nats.
    """
    # Normalize to 2D so the same code path handles scalar and multivariate inputs.
    C = np.atleast_2d(np.asarray(C, dtype=float).T).T
    E = np.atleast_2d(np.asarray(E, dtype=float).T).T

    assert C.shape[0] == E.shape[0], "C and E must have same number of samples"
    n = C.shape[0]
    assert 1 <= k < n, "k must satisfy 1 <= k < n"

    if standardize:
        C = StandardScaler().fit_transform(C)
        E = StandardScaler().fit_transform(E)

    CE = np.hstack([C, E])

    # KSG convention: use max norm / Chebyshev distance.
    joint_tree = KDTree(CE, metric="chebyshev", leaf_size=leaf_size)
    c_tree = KDTree(C, metric="chebyshev", leaf_size=leaf_size)
    e_tree = KDTree(E, metric="chebyshev", leaf_size=leaf_size)

    # Query k+1 because the closest point is the sample itself at distance 0.
    dist, _ = joint_tree.query(CE, k=k + 1)
    eps = dist[:, k]

    # KSG counts marginal neighbors strictly inside the joint kth-neighbor radius.
    # KDTree query_radius counts <= r, so move the radius infinitesimally downward.
    eps_strict = np.nextafter(eps, 0.0)

    # Counts include the point itself, so subtract 1.
    n_c = c_tree.query_radius(C, r=eps_strict, count_only=True) - 1
    n_e = e_tree.query_radius(E, r=eps_strict, count_only=True) - 1

    mi = digamma(k) + digamma(n) - np.mean(digamma(n_c + 1) + digamma(n_e + 1))

    # Finite-sample estimates can be slightly negative; the raw estimator is returned.
    return float(mi)


# ---------------------------------------------------------------------------
# Permutation-based bias correction wrapper
# ---------------------------------------------------------------------------

def mi_scalar_bias_corrected(C, E, n_permutations=200, mi_obs=None,
                              estimator=ksg_mi, verbose=False,
                              **estimator_kwargs):
    """
    MI with permutation-based finite-sample bias correction.

    Corrected MI: observed MI minus mean permuted MI. The result is an empirical
    excess-MI above an independence baseline, not a guaranteed unbiased estimate
    of the population MI.
    """
    n = len(C)

    if mi_obs is None:
        mi_obs = estimator(C, E, **estimator_kwargs)

    perm_mi = np.empty(n_permutations, dtype=float)

    for b in tqdm(range(n_permutations), disable=not verbose):
        perm = np.random.permutation(n)
        perm_mi[b] = estimator(C, E[perm], **estimator_kwargs)

    perm_mean = float(np.mean(perm_mi))
    mi_bc = float(mi_obs - perm_mean) 

    return mi_bc