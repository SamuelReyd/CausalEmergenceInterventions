from enum import Enum
from itertools import chain, combinations, product
from typing import Counter

import numpy as np
from tqdm import tqdm

# Utils
def coin(p):
    return np.random.rand() < p

def get_probas_from_pairs(C, E):
    vC, sC = np.unique(C, axis=0, return_inverse=True)
    vE, sE = np.unique(E, axis=0, return_inverse=True)

    # tabulate counts -> probabilities
    pc = np.bincount(sC).astype(float)
    pe = np.bincount(sE).astype(float)
    pj = np.bincount(sC * len(pe) + sE, 
                     minlength=len(pc)*len(pe)).astype(float) # Coocurence proba
    pj /= pj.sum()
    pc /= pc.sum()
    pe /= pe.sum()

    pi = np.outer(pc, pe).ravel()    
    return pc, pe, pj, pi, vC, vE

def pairs(X):
    return X[:,0], X[:,1]

def align_distributions(p1, p2, v1, v2):
    vals = np.union1d(v1, v2)
    a1 = np.zeros(len(vals), dtype=float)
    a2 = np.zeros(len(vals), dtype=float)
    a1[np.searchsorted(vals, v1)] = p1
    a2[np.searchsorted(vals, v2)] = p2
    return a1, a2, vals

def trajectory_to_pairs(X):
    return np.array([X[:-1],X[1:]]).transpose(1,0,2)

# Metrics
def KL(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def MI(Px, Py, Pj):
    Pi = np.outer(Px, Py).ravel()
    return KL(Pj, Pi)

# Data generation
def sample_with_parity(size, parity):
    if not size:
        return []
    x = np.random.randint(0, 2, size=size-1, dtype=int).tolist()
    x.append(parity^sum(x, 0)%2)
    np.random.shuffle(x)
    return x
    
def intervene(x, I):
    return [I.get(i, int(x[i])) for i in range(len(x))]
    
def sample_trajectory(u, fnt, confounder, N, I={}, n_runs=1, verbose=1, **kwargs):
    X = []
    for i in tqdm(range(n_runs), disable=not verbose or n_runs == 1):
        if callable(u):
            x = u(seed=i, **kwargs)
        else:
            x = u.copy()
        for _ in tqdm(range(N), disable=not verbose or n_runs > 1):
            _, x = fnt(x.copy(), I, confounder, **kwargs)
            X.append(x)
    X = trajectory_to_pairs(X) # -> from (2, N, d) to (N, 2, d)
    return X

def make_initial_conditions_from_prior(omega, prior, N):
    ids = np.random.choice(range(len(omega)), p=prior, size=N)
    return omega[ids]

def sample_pairs(cs, fnt, confounder, I={}, verbose=1, **kwargs):
    X = []
    for c in tqdm(cs, disable=not verbose):
        _, e = fnt(c.copy(), I, confounder, **kwargs)
        X.append((c,e))
    return np.array(X)

# Example sampling
def observe_parity(X):
    return np.array(X).sum(axis=-1) % 2

def sample_decoupling(x, I, confounder, gamma, **kwargs):
    c = intervene(x, I)
    if confounder:
        z = c[-1] # Z_t -> Z_t+1
        parity = z # Z_t -> X_t+1
        if coin(gamma): # Z_t -> Z_t+1
            z = 1 - z
    else:
        parity = sum(x) % 2 # V_t -> X_t+1
        if coin(gamma): # V_t -> V_t+1
            parity = 1 - parity
    e = sample_with_parity(len(c)-confounder, parity) # Macro -> Micro
    if confounder:
        e = e + [z]
    return c, e

def sample_downward(x, I, confounder, **kwargs):
    c = intervene(x, I)
    if confounder:
        # Ze is random, first bit of Xe is Zc, rest of Xe has parity of Ze
        zc = c[-1]
        ze = np.random.randint(2, dtype=int)
        e = [zc] + sample_with_parity(len(c) - 2, zc^ze) + [ze]
    else:
        # First bit is the parity, the rest are random
        e = [sum(c) % 2] + np.random.randint(2, size=len(x)-1, dtype=int).tolist()
    return c, e

def sample_down_dec(x, I, confounder, gamma, **kwargs):
    c = intervene(x, I)
    if confounder:
        zc = c[-1]
        ze = zc
        if coin(gamma): # V_t -> V_t+1
            ze = 1 - ze
        e = [zc] + sample_with_parity(len(c) - 2, zc^ze) + [ze]
    else:
        e1 = sum(c) % 2
        parity = e1
        if coin(gamma): # V_t -> V_t+1
            parity = 1 - parity
        e = [e1] + sample_with_parity(len(c)-1, e1^parity)
    return c, e

def sample_macro_random(x, I, **kwargs):
    c = I.get(0, x)
    e = np.random.randint(2, dtype=int)
    return c, e

def sample_macro_constant(x, I, gamma, **kwargs):
    c = I.get(0, x)
    e = c
    if coin(gamma):
        e = 1 - e
    return c, e

# Rosas-Jensen
def MI_from_data(C, E):
    pc, pe, pj, _, _, _ = get_probas_from_pairs(C, E)
    return MI(pc, pe, pj)

def Psi(C_m, E_m, C_M, E_M, verbose=False):
    I_macro = MI_from_data(C_M, E_M)
    I_upward = np.array([MI_from_data(C_m[:,i:i+1], E_M) for i in range(C_m.shape[1])])
    if verbose:
        print(f"Macro forward: {I_macro:.2f}")
        print(f"Upward: {I_upward.round(4)}")
    return I_macro - I_upward.sum()

def Delta(C_m, E_m, C_M, E_M, verbose=False):
    I_downward = np.array([MI_from_data(C_M, E_m[:,i:i+1]) for i in range(E_m.shape[1])])
    I_forward = np.array([
        sum([
            MI_from_data(C_m[:,j:j+1], E_m[:,i:i+1]) for j in range(E_m.shape[1])
            ]) 
            for i in range(E_m.shape[1])]
        )
    if verbose:
        print(f"Downward: {I_downward.round(2)}")
        print(f"Micro forward: {I_forward.round(2)}")
    return (I_downward - I_forward).max()

def Gamma(C_m, E_m, C_M, E_M, verbose=False):
    I_downward = np.array([MI_from_data(C_M, E_m[:,i:i+1]) for i in range(E_m.shape[1])])
    if verbose:
        print(f"Downward: {I_downward.round(2)}")
    return max(I_downward)

# Hoel-Albantakis
def get_TPM(pc, pe, pj):
    M = pj.reshape(len(pc), len(pe))   # M[c,e] = p(C=c, E=e)
    rows = M.sum(axis=1, keepdims=True)
    return np.divide(M, rows, out=np.zeros_like(M), where=rows != 0)

def EI_KL(C, E, verbose=False):
    pc, pe, pj, _, _, _ = get_probas_from_pairs(C, E)
    T = get_TPM(pc, pe, pj)
    kls = np.array([KL(Tc, pe) for Tc in T])
    if verbose: 
        print("T")
        print(T.round(2))
        print("pe:", pe.round(2))
        print("kls:", kls.round(2))
    return np.sum(pc * kls)

def EI_MI(C, E):
    pc, pe, pj, _, _, _ = get_probas_from_pairs(C, E)
    return MI(pc, pe, pj)

def CE_EI(C_m, E_m, C_M, E_M, verbose=0):
    ei_m = EI_KL(C_m, E_m, max(0, verbose-1))
    ei_M = EI_KL(C_M, E_M, max(0, verbose-1))
    if verbose:
        print(f"{ei_m=:.2f}")
        print(f"{ei_M=:.2f}")
    return ei_M - ei_m

# Main
def emergence(m, M, ei=True, psi=True, delta=True, gamma=True, verbose=1):
    if ei:
        dei = CE_EI(*pairs(m), *pairs(M), verbose)
        print(f"  ---> ΔEI = {dei:.2f}")
        print()
    if psi:
        psi = Psi(*pairs(m), *pairs(M), verbose)
        print(f"  ---> {psi=:.2f}")
        print()
    if delta:
        delta = Delta(*pairs(m), *pairs(M), verbose)
        print(f"  ---> {delta=:.2f}")
        print()
    if gamma:
        gamma = Gamma(*pairs(m), *pairs(M), verbose)
        print(f"  ---> {gamma=:.2f}")
        print()

def show_emergence_run(sampling, confounder, prior, D, X, Z, V, **kwargs):
        t = f" {sampling=}, {confounder=}, {prior=} "
        print(f"{t:=^100}")
        emergence(X, V, **kwargs)
        print()

def show_emergence_runs(runs, ei=True, psi=True, delta=True, gamma=True, verbose=1):
    for (sampling, confounder, prior), (D,X,Z,V) in runs.items():
        show_emergence_run(sampling, confounder, prior, D, X, Z, V, ei=ei, psi=psi, delta=delta, gamma=gamma, verbose=verbose)

def get_prior(d, cg_fnt=None):
    omega_m = np.array(list(product((0,1), repeat=d)))
    pc_m = np.full(len(omega_m), 1/len(omega_m))
    if cg_fnt is None:
        return omega_m, pc_m, None
    pc_M = pc_m.copy()
    f = cg_fnt(omega_m).astype(bool)
    pc_M[f] /= pc_m[f].sum() * 2
    pc_M[~f] /= pc_m[~f].sum() * 2
    return omega_m, pc_m, pc_M

def coarse_grain(m, cg_fnt, confounder, test_confounder):
    m = np.array(m)
    if confounder and test_confounder:
        return m[..., -1]
    elif confounder:
        return cg_fnt(m[...,:-1])
    return cg_fnt(m)

def get_runs(N, d, sample_fnt, cg_fnt, **kargs):
    runs = {}
    # for sampling, confounder, prior in product(("traj", "pairs"), (False, True), (None, "traj", "uniform-micro", "uniform-macro")):
    for sampling, confounder, prior in product(("traj", "pairs"), (False, True), (None, "uniform-micro")):
        if sampling == "traj" and prior is not None:
            continue
        if sampling != "traj" and prior is None:
            continue
        print(sampling, confounder, prior)
        if sampling == "traj":
            u = np.ones(d, dtype=int).tolist()
            D = sample_trajectory(u + [sum(u)%2] * confounder, sample_fnt, confounder, N, **kargs)
        else:
            omega, pc_m, pc_M = get_prior(d + confounder, cg_fnt)
            if prior == "traj":
                D = runs[("traj", confounder, None)][1-confounder]
                pc, _, _, _, omega, _ = get_probas_from_pairs(*pairs(D))
            elif prior == "uniform-micro":
                omega, pc, _ = get_prior(d + confounder, cg_fnt)
            else:
                omega, _, pc = get_prior(d + confounder, cg_fnt)
            cs = make_initial_conditions_from_prior(omega, pc, N)
            D = sample_pairs(cs, sample_fnt, confounder, **kargs)
        if confounder:
            D, X, Z = D, D[...,:-1], D[...,-1]
            V = cg_fnt(X)
        else:
            D, X, Z = None, D, None
            V = cg_fnt(X)
        runs[(sampling, confounder, prior)] = [D, X, Z, V]
    return runs

if __name__ == "__main__":
    N = 100_000
    d = 10
    gamma = .01

    runs_decoupling = get_runs(N, d, sample_decoupling, observe_parity, gamma=gamma)
    # runs_downward = get_runs(N, d, sample_downward, observe_parity)
    # runs_down_dec = get_runs(N, d, sample_down_dec, observe_parity, gamma=gamma)

    show_emergence_runs(runs_decoupling, ei=False, delta=False, gamma=False, verbose=0)
    # show_emergence_runs(runs_downward, ei=False, delta=False, gamma=False)
    # show_emergence_runs(runs_down_dec, ei=False, delta=False, gamma=False)

