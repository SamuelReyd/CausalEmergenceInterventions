import numpy as np
from tqdm import tqdm

pi = np.pi
WIDTH = 500

## Simulation
def angle(vects):
    """
    Signed angle of 2-D vectors.
    vects : (..., 2)  →  (...,)  in [0, 2π)
    """
    return (np.arctan2(vects[..., 1], vects[..., 0]) + 2 * pi) % (2 * pi)
    
def direction(a):
    """
    Unit vectors from angles.
    a : (...,)  →  (..., 2)
    """
    return np.stack([np.cos(a), np.sin(a)], axis=-1)
    
def _eff_pairwise_dist(pos):
    # Squared norms: (B, N)
    sq_norms = (pos ** 2).sum(axis=-1)
 
    # ‖xᵢ - xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² - 2 xᵢ·xⱼ
    # sq_norms[:, :, None] : 
    # sq_norms[:, None, :] :
    # pos @ pos.swapaxes(-1,-2) :   ← batched matmul, no extra dim
    sq_dist = (
        sq_norms[:, :, None] # (B, N, 1)
        + sq_norms[:, None, :] #  (B, 1, N)
        - 2 * (pos @ pos.swapaxes(-1, -2)) # (B, N, N)
    )

    # Numerical noise can make same-boid distance slightly negative
    np.clip(sq_dist, 0, None, out=sq_dist)

    return sq_dist

def get_filters(s, close_r, regular_r):
    """
    Parameters
    ----------
    s         : (B, N, 3)
    close_r   : scalar close radius per run
    regular_r : scalar regular radius per run
 
    Returns
    -------
    close_mask   : (B, N, N)  bool
    regular_mask : (B, N, N)  bool
    """
    pos = s[:, :, :2]                                     # (B, N, 2)
    sq_dist = _eff_pairwise_dist(pos)
    
    not_self = ~np.eye(s.shape[1], dtype=bool)[None]               # (1, N, N)
 
    # Compare squared distances to squared radii — no sqrt needed
    close_mask = (sq_dist <= close_r ** 2)   & not_self   # (B, N, N)
 
    has_close    = close_mask.any(axis=2, keepdims=True)  # (B, N, 1)
    regular_mask = (
        (sq_dist <= regular_r ** 2)
        & not_self
        & ~close_mask
        & ~has_close
    )                                                      # (B, N, N)
 
    return close_mask, regular_mask
    
def _safe_mean_pos(mask, pos):
    """
    Weighted mean position of neighbours.
    mask : (B, N, N),  pos : (B, N, 2)  →  (B, N, 2),  NaN where count == 0
    """
    counts = mask.sum(axis=2, keepdims=True)          # (B, N, 1)
    sums   = mask @ pos                               # (B, N, 2)  — batched matmul
    with np.errstate(invalid='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)
        
def _safe_mean_scalar(mask, values):
    """
    Weighted mean of a scalar field over neighbours.
    mask : (B, N, N),  values : (B, N)  →  (B, N),  NaN where count == 0
    """
    counts = mask.sum(axis=2)                         # (B, N)
    sums   = (mask * values[:, None, :]).sum(axis=2)  # (B, N)
    with np.errstate(invalid='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)
        
def separation(s, close_mask):
    mean = _safe_mean_pos(close_mask, s[:, :, :2])   # (B, N, 2)
    return angle(-(mean - s[:, :, :2]))              # (B, N)
    
def cohesion(s, regular_mask):
    mean = _safe_mean_pos(regular_mask, s[:, :, :2])
    return angle(mean - s[:, :, :2])                 # (B, N)
    
def alignment(s, regular_mask):
    return _safe_mean_scalar(regular_mask, s[:, :, 2])  # (B, N)
    
def turn_towards(headings, target, max_turn, noise_std):
    """
    Rotate headings toward target by at most max_turn, shortest-arc.
 
    headings  : (B, N)
    target    : (B, N)   NaN where no neighbour → no change
    max_turn  : scalar value
    noise_std : scalar value
 
    Returns updated headings (B, N).
    """
    delta = target - headings                                      # (B, N)
    # Wrap into (-π, π]
    delta = (delta + pi) % (2 * pi) - pi
 
    # Optional per-run noise — only draw random numbers when needed
    if noise_std > 0:
        delta += np.random.normal(0, noise_std, size=delta.shape)
 
    step     = np.sign(delta) * np.minimum(np.abs(delta), max_turn)
    new_h    = (headings + step) % (2 * pi)
 
    # Where target is NaN (no neighbour), keep original heading unchanged
    return np.where(np.isnan(target), headings, new_h)
    
def apply_turns(s, close_mask, regular_mask, hp):
    """
    params : dict of (B,) arrays for each key in PARAM_KEYS
    """
 
    h = s[:, :, 2].copy()                         # (B, N)
    h = turn_towards(h, separation(s, close_mask),  hp["separation_factor"], hp["noise"])
    h = turn_towards(h, cohesion(s, regular_mask),  hp["cohesion_factor"], hp["noise"])
    h = turn_towards(h, alignment(s, regular_mask), hp["alignment_factor"], hp["noise"])
    return h
    
def update_flocks_batch(s, hp):
    """
    One Euler step for B runs simultaneously.
 
    s      : (B, N, 3)
    hp     : dict of parameters
 
    Returns new state (B, N, 3).
    """
    # Reshape radii and speed for broadcasting
    close_mask, regular_mask = get_filters(s, hp["minimum_separation"], hp["view_radius"])
    if hp["confounder"]:
        regular_mask = np.zeros_like(regular_mask)
        regular_mask[:,:,0] = 1
 
    new_h   = apply_turns(s, close_mask, regular_mask, hp)  # (B, N)
    new_pos = s[:, :, :2] + direction(new_h) * hp["max_speed"]           # (B, N, 2)
    # TEST: what if we wake it toric again?
    new_pos[:,:,:2] = new_pos[:,:,:2] % WIDTH
 
    return np.concatenate([new_pos, new_h[:, :, None]], axis=-1)
    
def make_sweep(boids, hp, verbose=False):
    """
    Run B simulations in parallel using pure NumPy.
 
    Parameters
    ----------
    boids   : (B, N, 3)  shared initial state
    hp : dict with the configurations
 
    Returns
    -------
    histories : (B, n_steps+1, N, 3)
    """
 
    # Tile initial state across batch dimension: (B, N, 3)
    if len(boids.shape) == 2:
        s = boids[None, ...]
    else:
        s = boids[:]

    
    B       = s.shape[0]
    n_steps = hp["n_steps"]
 
    # Pre-allocate history array to avoid repeated list appending
    N = s.shape[1]
    history = np.empty((B, n_steps + 1, N, 3), dtype=np.float64)
    history[:, 0] = s
 
    for t in tqdm(range(n_steps), disable=not verbose, desc="Simulating trajectories"):
        s = update_flocks_batch(s, hp)
        history[:, t + 1] = s
 
    return history
    
def make_run_flocking(boids, params, noise, n_steps):
    """Single-run convenience wrapper, returns (n_steps+1, N, 3)."""
    history = make_sweep(boids, np.array(params)[None,:], noise, n_steps)
    return history[0]
  
## Initialization
def grid_coords(p, s):
    return int(p[0] / s), int(p[1] / s)
    
def fits(p, grid, r, s, w, h):
    gx, gy = grid_coords(p, s)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            ngx, ngy = gx + dx, gy + dy
            if 0 <= ngx < w and 0 <= ngy < h:
                neighbor = grid[ngy][ngx]
                if neighbor is not None and np.linalg.norm(np.array(p) - np.array(neighbor)) < r:
                    return False
    return True
    
def generate_poisson_disc_samples(width, height, r, k=30):
    s = r / np.sqrt(2)
    w = int(np.ceil(width / s))
    h = int(np.ceil(height / s))
    grid = [[None for _ in range(w)] for _ in range(h)]

    process_list = []
    sample_points = []

    # Generate the first sample
    first_point = (np.random.uniform(0, width), np.random.uniform(0, height))
    process_list.append(first_point)
    sample_points.append(first_point)
    gx, gy = grid_coords(first_point, s)
    grid[gy][gx] = first_point

    while process_list:
        idx = np.random.randint(0, len(process_list))
        parent_point = process_list[idx]
        found = False

        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            candidate = np.array(parent_point) + direction * np.random.uniform(r, 2 * r)

            if 0 <= candidate[0] < width and 0 <= candidate[1] < height and \
                    fits(candidate, grid, r, s, w, h):
                process_list.append(tuple(candidate))
                sample_points.append(tuple(candidate))
                gx, gy = grid_coords(candidate, s)
                grid[gy][gx] = tuple(candidate)
                found = True
                break

        if not found:
            process_list.pop(idx)

    return np.array(sample_points)
    
def init_boids(hp, base_range=50, k=30, min_sep=8.4, pad=5, seed=None, init_angle_range=pi/5, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    # Generate points 
    points = generate_poisson_disc_samples(base_range, base_range, min_sep, k)
    points -= np.array([base_range, base_range]) / 2
    ids = np.argsort(np.linalg.norm(points, axis=1))
    if len(points) < hp["n_boids"]:
        print(f"Warning: Not enough boids generated with {base_range=}, returning only {len(points)} boids")
    else:
        points = points[ids[:hp["n_boids"]]]

    cx = np.random.uniform(hp["width"]/12, 11*hp["width"]/12)
    cy = np.random.uniform(2*hp["height"]/3, hp["height"])
    center = np.array([cx, cy])
    points += center
    
    boids = np.zeros((hp["n_boids"], 3))
    boids[:,:2] = points
    boids[:, 2] = np.random.uniform(
        3*np.pi / 2 - init_angle_range, 
        3*np.pi / 2 + init_angle_range, 
        size=hp["n_boids"])
    
    return boids

def make_large_layout(n_boids, min_sep=8.4, k=30, oversample=8, seed=None):
    """
    Parameters
    ----------
    n_boids    : int, number of boids per run
    min_sep    : float, minimum pairwise distance
    k          : int, Poisson disc rejection attempts
    oversample : int, target M = oversample * n_boids total points.
                 Higher values give more selection variety but cost more
                 to generate. 5-10 is a good range.
    seed       : optional int
 
    Returns
    -------
    layout : (M, 2)  all generated points, centred at origin.
    """
    if seed is not None:
        np.random.seed(seed)
 
    M_target = oversample * n_boids
 
    # Area needed: Poisson disc packs roughly 1 point per (r^2 * sqrt(3)/2)
    # area. We add a 40% margin to reliably hit M_target.
    area_per_point = (min_sep ** 2) * np.sqrt(3) / 2
    side = np.sqrt(M_target * area_per_point * 1.4)
 
    points = generate_poisson_disc_samples(side, side, min_sep, k)
 
    if len(points) < n_boids:
        raise ValueError(
            f"Large layout only produced {len(points)} points, need at least "
            f"{n_boids}. Decrease min_sep or increase oversample."
        )
 
    points -= points.mean(axis=0)   # centre at origin
    return points                   # (M, 2)

def init_boids_batch(hp, B, init_angle_range=pi/5, seed=None):
    """
    Parameters
    ----------
    hp               : dict with keys n_boids, width, height
    B                : int, number of runs
    layout           : (M, 2) from make_large_layout, centred at origin
    init_angle_range : half-width of heading distribution (radians)
    seed             : optional int
 
    Returns
    -------
    states : (B, N, 3)
    """
    if seed is not None:
        np.random.seed(seed)
 
    N = hp["n_boids"]

    layout = make_large_layout(N, min_sep=hp["minimum_separation"], k=30, oversample=20)
 
    # --- B random selection centres within the layout bounds ---------------
    lo = layout.min(axis=0)
    hi = layout.max(axis=0)
    centres = np.random.uniform(lo, hi, size=(B, 2))          # (B, 2)
 
    # --- Vectorized nearest-N selection ------------------------------------
    # Squared distances from every centre to every layout point: (B, M)
    diff  = layout[None, :, :] - centres[:, None, :]          # (B, M, 2)
    sq_dists = (diff ** 2).sum(axis=-1)                       # (B, M)
 
    # argpartition gives the N smallest indices (unordered) in O(M) per row,
    # cheaper than a full argsort which would be O(M log M)
    nearest = np.argpartition(sq_dists, N, axis=1)[:, :N]     # (B, N)
 
    # Gather positions via advanced indexing: (B, N, 2)
    local_positions = layout[nearest]
 
    # Re-centre each cluster at its own centroid so arena translation is clean
    local_positions -= local_positions.mean(axis=1, keepdims=True)
 
    # --- Random arena positions --------------------------------------------
    cx = np.random.uniform(0, WIDTH, size=B)
    cy = np.random.uniform(0, WIDTH, size=B)
    arena_centres = np.stack([cx, cy], axis=-1)               # (B, 2)
 
    positions = local_positions + arena_centres[:, None, :]   # (B, N, 2)
 
    # --- Random headings ---------------------------------------------------
    base_heading = np.random.uniform(0, 2*pi, size=B)
    headings = np.random.uniform(
        3*pi/2 - init_angle_range,
        3*pi/2 + init_angle_range,
        size=(B, N)
    )
    headings = headings + np.tile(base_heading, (headings.shape[1],1)).T
    headings = headings % (2*pi)
 
    return np.concatenate([positions, headings[:, :, None]], axis=-1)  # (B, N, 3)