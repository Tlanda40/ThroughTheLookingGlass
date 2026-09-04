"""
cage_relative_msd.py

Compute the average root-mean-square displacement of each particle relative
to the average displacement of its six nearest neighbors ("cage-relative"
or "non-affine" displacement), from a linked-trajectories CSV produced by
detect_and_link.py (columns: frame, x, y, particle, ...).

This quantity subtracts out any bulk/collective motion shared by a
particle's local neighborhood (e.g. drift, convection, or affine
deformation of the whole sample) and isolates genuine relative
("cage-breaking") motion -- the standard approach in colloidal glass /
jamming studies (cf. Weeks & Weitz, Science 2000).

For each lag time dt (in frames) and each usable time origin t:
    1. Take particle positions at frame t and frame t+dt.
    2. For every particle present in both frames, compute its raw
       displacement vector d_i = r_i(t+dt) - r_i(t).
    3. Find its 6 nearest neighbors using positions at frame t.
    4. Compute the cage-relative displacement:
           d_i_rel = d_i - mean(d_j for j in the 6 nearest neighbors)
       (only neighbors that also have a valid displacement are averaged --
       neighbors are found from ALL particles at frame t, but only those
       present in both t and t+dt actually get compared)
    5. Accumulate |d_i_rel|^2 into a running sum for that lag dt.

The output is:
    - a CSV with one row per lag time: dt (frames), dt (seconds, if FPS
      given), MSD_rel (mean squared cage-relative displacement),
      RMSD_rel (its square root), and the number of (particle, origin)
      samples used.
    - a single overall scalar: the RMSD_rel averaged across all sampled
      lag times, if you just want one number.
    - an optional log-log plot of RMSD_rel vs. lag time.

NOTE ON SCALE: with ~105,000 trajectories over 4000 frames, computing
every possible time origin for every lag time is not feasible. This
script subsamples time origins per lag (N_ORIGINS_PER_LAG) and lets you
choose which lag times to evaluate (LAG_FRAMES). Increase these for a
more precise average at the cost of runtime.

NOTE ON BOUNDARIES: this does NOT apply periodic boundary conditions.
If your sample has periodic boundaries, both the nearest-neighbor search
and the displacement calculation need minimum-image convention -- ask
if you need that added.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
# 1. CONFIG -- edit these values
# ----------------------------------------------------------------------

CSV_PATH = "/Volumes/Expansion/recordings/linked_trajectories.csv"
OUTPUT_DIR = os.path.dirname(CSV_PATH)

# Which lag times (in frames) to evaluate. dt=1 is the smallest and most
# direct estimate of relative motion; add larger lags if you want an
# MSD-style curve showing how relative displacement grows with time.
LAG_FRAMES = [1, 2, 5, 10, 20, 50]

# How many distinct time origins to sample per lag time. Set to None to
# use every possible origin (n_frames - dt of them) -- can be slow for
# large datasets.
N_ORIGINS_PER_LAG = 200

# Number of nearest neighbors to average over (excludes the particle
# itself).
K_NEIGHBORS = 6

# Frames per second, if you want lag times reported in real time units
# as well as frames. Set to None to skip.
FPS = None

RANDOM_SEED = 0


def load_trajectories(path):
    """Load linked trajectories and index by frame for fast lookup."""
    df = pd.read_csv(path, usecols=["frame", "x", "y", "particle"])
    df["frame"] = df["frame"].astype(int)
    df["particle"] = df["particle"].astype(int)

    frames = sorted(df["frame"].unique())
    by_frame = {}
    for fr, sub in df.groupby("frame"):
        by_frame[fr] = sub.set_index("particle")[["x", "y"]]
    return by_frame, frames


def cage_relative_sq_displacements(pos_t, pos_t_dt, k=6):
    """
    Given position tables (index=particle, columns=[x,y]) at frame t and
    frame t+dt, return an array of squared cage-relative displacements,
    one per particle present in both frames with enough neighbors.
    """
    common = pos_t.index.intersection(pos_t_dt.index)
    if len(common) < k + 2:
        return np.array([])

    p0 = pos_t.loc[common].to_numpy()
    p1 = pos_t_dt.loc[common].to_numpy()
    disp = p1 - p0  # raw displacement vectors, shape (N, 2)

    tree = cKDTree(p0)
    # k+1 because the particle itself (distance 0) is always the closest
    _, idx = tree.query(p0, k=k + 1)

    neighbor_idx = idx[:, 1:]  # drop self
    neighbor_disp_mean = disp[neighbor_idx].mean(axis=1)

    rel_disp = disp - neighbor_disp_mean
    sq_disp = np.sum(rel_disp ** 2, axis=1)
    return sq_disp


def compute_cage_relative_msd(by_frame, frames, lag_frames, k=6,
                               n_origins_per_lag=None, seed=0):
    rng = np.random.default_rng(seed)
    n_frames = len(frames)

    results = []
    for dt in lag_frames:
        possible_origin_idxs = np.arange(0, n_frames - dt)
        if len(possible_origin_idxs) == 0:
            print(f"  lag {dt}: no valid origins, skipping.")
            continue

        if n_origins_per_lag is not None and n_origins_per_lag < len(possible_origin_idxs):
            chosen = rng.choice(possible_origin_idxs, size=n_origins_per_lag, replace=False)
        else:
            chosen = possible_origin_idxs

        all_sq = []
        for oi in chosen:
            t0 = frames[oi]
            t1 = frames[oi + dt]
            if t0 not in by_frame or t1 not in by_frame:
                continue
            sq = cage_relative_sq_displacements(by_frame[t0], by_frame[t1], k=k)
            if sq.size:
                all_sq.append(sq)

        if not all_sq:
            print(f"  lag {dt}: no usable particle pairs, skipping.")
            continue

        all_sq = np.concatenate(all_sq)
        msd_rel = all_sq.mean()
        rmsd_rel = np.sqrt(msd_rel)
        results.append({
            "lag_frames": dt,
            "lag_seconds": dt / FPS if FPS else np.nan,
            "n_samples": all_sq.size,
            "msd_rel": msd_rel,
            "rmsd_rel": rmsd_rel,
        })
        print(f"  lag {dt} frames: RMSD_rel = {rmsd_rel:.4f} px "
              f"(from {all_sq.size} particle-origin samples)")

    return pd.DataFrame(results)


def main():
    print(f"Loading trajectories from:\n  {CSV_PATH}")
    by_frame, frames = load_trajectories(CSV_PATH)
    print(f"Loaded {len(frames)} frames, "
          f"{sum(len(v) for v in by_frame.values())} total detections.")

    print(f"\nComputing cage-relative RMSD "
          f"(k={K_NEIGHBORS} nearest neighbors) for lag times: {LAG_FRAMES}")
    results_df = compute_cage_relative_msd(
        by_frame, frames, LAG_FRAMES, k=K_NEIGHBORS,
        n_origins_per_lag=N_ORIGINS_PER_LAG, seed=RANDOM_SEED,
    )

    if results_df.empty:
        print("No results computed -- check your data and LAG_FRAMES.")
        return

    out_csv = os.path.join(OUTPUT_DIR, "cage_relative_msd.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-lag results to:\n  {out_csv}")

    overall_rmsd = np.sqrt(results_df["msd_rel"].mean())
    print(f"\nOverall average RMSD relative to {K_NEIGHBORS} nearest "
          f"neighbors (averaged across sampled lag times): "
          f"{overall_rmsd:.4f} px")

    # ---- optional plot ----
    try:
        import matplotlib.pyplot as plt
        x = results_df["lag_seconds"] if FPS else results_df["lag_frames"]
        xlabel = "Lag time (s)" if FPS else "Lag time (frames)"
        plt.figure(figsize=(6, 4))
        plt.plot(x, results_df["rmsd_rel"], "o-")
        plt.xlabel(xlabel)
        plt.ylabel("Cage-relative RMSD (px)")
        plt.title(f"Cage-relative RMSD vs lag time (k={K_NEIGHBORS} neighbors)")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "cage_relative_msd.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to:\n  {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()