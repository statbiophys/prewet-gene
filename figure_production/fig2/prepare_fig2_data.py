#!/usr/bin/env python3
"""Reduce the lattice runs to the tables of Fig. 2, in processed_data/.

    python figure_production/fig2/prepare_fig2_data.py [--data-root DIR] [--date YYYYMMDD] [--out DIR]

Reads   <data-root>/<date>/proximal_promoter_transc_d_50_c_<c>/output_cpu<k>.json
Writes  fig2_concentration_summary.csv, fig2_hist_local_concentration.csv,
        fig2_hist_pattern_distance.csv, fig2_transcription_bootstrap.csv, fig2_scalars.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.lattice_runs import load_sweeps    # figure_production/common/


CONCS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
         0.009, 0.010, 0.011, 0.012, 0.013, 0.014,
         0.015, 0.016, 0.017, 0.018]
NCONC = len(CONCS)
NRUNS = 16
EXP = "proximal_promoter_transc_d_50_c_{:.3f}"

# threshold
THRESH_CLOSED = 6.5   # pattern_distance < thresh  <=>  "closed"

# bin edges
YE_LOCAL_CONC = (np.arange(28) - 0.5) / 26.0      # the k/26 lattice grid
N_BINS_DIST = 70                                  # edges; 69 bins
DIST_PERCENTILE = 99.5                            # upper edge of the distance axis

NBOOT = 100           # bootstrap replicates


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reduce the raw runs to the tables Fig. 2 needs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default=None,
                   help="directory holding the dated run folders "
                        "(default: <repo>/data)")
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)),
                   help="a directory at or below the repository root")
    p.add_argument("--date", default=None,
                   help="run-date folder under --data-root (default: the "
                        "latest one holding a lattice sweep)")
    p.add_argument("--out", default=None,
                   help="output directory (default: figure_production/processed_data)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the run-level bootstrap, which is unseeded "
                        "in the notebook; -1 leaves the global RNG untouched")
    return p.parse_args(argv)


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def resolve_repo(repo):
    """The repository root: the directory holding Project.toml, at or above
    `repo` (by default this script's directory)."""
    d = os.path.abspath(os.path.expanduser(repo))
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "Project.toml")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("no Project.toml at or above {}; pass --repo".format(repo))


def latest_lattice_date(data_root):
    """The newest dated folder under data_root holding a lattice sweep
    (proximal_promoter_transc_d_50_c_*), as written by run_lattice_sweep.sh."""
    import glob
    dates = sorted({os.path.basename(os.path.dirname(p)) for p in
                    glob.glob(os.path.join(data_root, "*", "proximal_promoter_transc_d_50_c_*"))})
    if not dates:
        raise SystemExit("no lattice runs under {} -- run figure_production/run_lattice_sweep.sh "
                         "or pass --data-root / --date".format(data_root))
    return dates[-1]

def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------

def load_runs(repo, data_root, date):
    """Load all runs of the sweep."""

    src = os.path.join(os.path.abspath(os.path.expanduser(data_root)), date)
    if not os.path.isdir(src):
        raise SystemExit("no such data directory: {}\n"
                         "pass --data-root (the parent of {}/)".format(src, date))


    log("reading {} concentrations x {} runs from {} ...".format(
        NCONC, NRUNS, src))
    t0 = time.time()
    runs = load_sweeps([os.path.join(src, EXP.format(c)) for c in CONCS], NRUNS,
                       ("off_times", "pattern_distances", "transcription_activity",
                        "local_concentration"))
    off_times = runs["off_times"]
    log("read in {:.0f} s".format(time.time() - t0))

    # rescale the event times to MC sweeps
    params = np.array(CONCS)
    total_vol_frac = (params + 0.02) * 62 ** 3
    sampling_multiplier = 50000 / total_vol_frac
    for i in range(NCONC):
        for k in range(NRUNS):
            off_times[i][k] = off_times[i][k] * sampling_multiplier[i]

    return dict(off_times=off_times,
                pattern_distances=runs["pattern_distances"],
                transcription_activity=runs["transcription_activity"],
                local_concentration=runs["local_concentration"],
                src=src)


# ----------------------------------------------------------------------------
# reduce
# ----------------------------------------------------------------------------

def split_closed_open(values, dists, cidx, ridx, thresh=THRESH_CLOSED, closed=True):
    """Values of one observable in one run,
    restricted to the frames where the probe is closed (or open)."""
    v = np.array(values[cidx][ridx])
    d = np.array(dists[cidx][ridx])
    idx = np.where(d < thresh) if closed else np.where(d >= thresh)
    return idx, v[idx]


def reduce_runs(R, seed):
    """Summaries, bootstrap and histograms of the sweep."""
    dists = R["pattern_distances"]
    conc = R["local_concentration"]
    transc = R["transcription_activity"]

    # ---- theta ------------------------------------------------------------
    theta = 40 * 1e3 / (6e23 * (25 * 1e-9) ** 3)      # volume fraction -> uM

    # ---- conditional means and occupancies ----------------------------
    phi_local_closed = np.zeros(NCONC)
    phi_local_open = np.zeros(NCONC)
    p_closed = np.zeros(NCONC)
    p_open = np.zeros(NCONC)

    for c in range(NCONC):
        pooled_closed, pooled_open = [], []
        n_closed = n_open = n_total = 0
        for r in range(NRUNS):
            ic, vc = split_closed_open(conc, dists, c, r, closed=True)
            io, vo = split_closed_open(conc, dists, c, r, closed=False)
            pooled_closed = np.concatenate((pooled_closed, vc))
            pooled_open = np.concatenate((pooled_open, vo))
            n_closed += len(ic[0])
            n_open += len(io[0])
            n_total += len(conc[c][r])
        phi_local_closed[c] = np.mean(pooled_closed)
        phi_local_open[c] = np.mean(pooled_open)
        p_closed[c] = n_closed / n_total
        p_open[c] = n_open / n_total

    transcription_closed = np.zeros(NCONC)
    transcription_open = np.zeros(NCONC)
    for c in range(NCONC):
        for arr, is_closed in ((transcription_closed, True),
                               (transcription_open, False)):
            arr[c] = np.mean(np.concatenate([
                split_closed_open(transc, dists, c, r, closed=is_closed)[1]
                for r in range(NRUNS)]))

    # ---- run-level (cluster) bootstrap --------------------------------
    if seed >= 0:
        np.random.seed(seed)
        log("bootstrap seeded with {}".format(seed))
    else:
        log("bootstrap left unseeded (as in the notebook)")

    boot_closed = np.zeros((NBOOT, NCONC))
    boot_open = np.zeros((NBOOT, NCONC))
    for i in range(NBOOT):
        runs_ = np.random.randint(0, NRUNS, size=(NCONC, NRUNS))
        for c in range(NCONC):
            for arr, is_closed in ((boot_closed, True), (boot_open, False)):
                pooled = np.concatenate([
                    split_closed_open(transc, dists, c, r, closed=is_closed)[1]
                    for r in runs_[c]])
                arr[i, c] = np.mean(pooled) if len(pooled) else np.nan
    log("bootstrap done ({} replicates)".format(NBOOT))

    # ---- the one kinetic input the figure needs -----------------------
    mean_toff_rnap = float(np.mean(
        [np.mean(np.concatenate(R["off_times"][k])) for k in range(NCONC)]))

    # ---- pool across runs, then bin -----------------------------------
    def pool(data):
        out = [np.concatenate([np.ravel(data[c][r]) for r in range(NRUNS)])
               for c in range(NCONC)]
        return [x[np.isfinite(x)] for x in out]

    pooled_transc, pooled_conc, pooled_dist = pool(transc), pool(conc), pool(dists)

    # the distance axis runs to a percentile of the pooled data, so the cutoff
    # is frozen here and recorded in scalars.json
    dist_hi = float(np.percentile(np.concatenate(pooled_dist), DIST_PERCENTILE))
    ye_dist = np.linspace(0.0, dist_hi, N_BINS_DIST)

    def histogram_matrix(pooled, edges):
        return np.column_stack([
            np.histogram(p, bins=edges, density=True)[0] for p in pooled])

    H_conc = histogram_matrix(pooled_conc, YE_LOCAL_CONC)
    H_dist = histogram_matrix(pooled_dist, ye_dist)

    params = np.array(CONCS)
    return dict(
        theta=theta, mean_toff_rnap=mean_toff_rnap, dist_hi=dist_hi,
        summary=pd.DataFrame({
            "c_phi": params,
            "c_uM": params * theta,
            "phi_local_closed": phi_local_closed,
            "phi_local_open": phi_local_open,
            "p_closed": p_closed,
            "p_open": p_open,
            "transcription_closed": transcription_closed,
            "transcription_open": transcription_open,
            "mean_transcription": [p.mean() for p in pooled_transc],
            "mean_local_concentration": [p.mean() for p in pooled_conc],
            "mean_pattern_distance": [p.mean() for p in pooled_dist],
            "n_samples": [p.size for p in pooled_dist],
        }),
        hist_conc=hist_frame(H_conc, YE_LOCAL_CONC, params),
        hist_dist=hist_frame(H_dist, ye_dist, params),
        bootstrap=bootstrap_frame(boot_closed, boot_open, params),
    )


def hist_frame(H, edges, params):
    """bins x concentrations, carrying both bin edges so the notebook can
    rebuild the pcolormesh grid without recomputing the percentile cutoff."""
    df = pd.DataFrame(H, columns=["c_{:.3f}".format(c) for c in params])
    df.insert(0, "bin_right", edges[1:])
    df.insert(0, "bin_left", edges[:-1])
    return df


def bootstrap_frame(boot_closed, boot_open, params):
    cols = ["c_{:.3f}".format(c) for c in params]
    a = pd.DataFrame(boot_closed, columns=cols)
    a.insert(0, "state", "closed")
    b = pd.DataFrame(boot_open, columns=cols)
    b.insert(0, "state", "open")
    out = pd.concat([a, b], ignore_index=True)
    out.insert(0, "replicate", list(range(NBOOT)) * 2)
    return out


# ----------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------

def write_outputs(D, out_dir, args, repo, src):
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "fig2_concentration_summary.csv": D["summary"],
        "fig2_hist_local_concentration.csv": D["hist_conc"],
        "fig2_hist_pattern_distance.csv": D["hist_dist"],
        "fig2_transcription_bootstrap.csv": D["bootstrap"],
    }
    # Full round-trip precision: the cross-validation that picks the smoothing
    # weight in panel (d) is ill-conditioned, and a rounded input selects a
    # different fit.
    for name, df in paths.items():
        df.to_csv(os.path.join(out_dir, name), index=False,
                  float_format="%.17g")

    scalars = {
        "theta": D["theta"],
        "theta_units": "uM per unit volume fraction",
        "mean_toff_rnap": D["mean_toff_rnap"],
        "K_D_bands": 2.3e-3,
        "K_D_curves": 2.33e-3,
        "phi_R": 0.02,
        "threshold_closed": THRESH_CLOSED,
        "dist_bin_hi": D["dist_hi"],
        "dist_bin_percentile": DIST_PERCENTILE,
        "dist_n_edges": N_BINS_DIST,
        "local_conc_bin_note": "edges = (arange(28) - 0.5)/26, the k/26 lattice grid",
        "n_conc": NCONC,
        "n_runs": NRUNS,
        "n_bootstrap": NBOOT,
        "bootstrap_seed": args.seed,
        "source_data": src,
        "source_date_folder": args.date,
        "repo_commit": git_commit(repo),
        "generated_by": os.path.basename(__file__),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "fig2_scalars.json"), "w") as f:
        json.dump(scalars, f, indent=2)
        f.write("\n")

    total = 0
    for name in list(paths) + ["fig2_scalars.json"]:
        p = os.path.join(out_dir, name)
        n = os.path.getsize(p)
        total += n
        print("  {:32s} {:>8.1f} KB".format(name, n / 1024))
    print("  {:32s} {:>8.1f} KB".format("TOTAL", total / 1024))


def main(argv=None):
    args = parse_args(argv)
    repo = resolve_repo(args.repo)
    args.data_root = args.data_root or os.path.join(repo, "data")
    args.date = args.date or latest_lattice_date(args.data_root)
    log("lattice runs: {}/{}".format(args.data_root, args.date))
    # processed_data/ is shared by the figures, one level up from this
    # script: figure_production/processed_data/
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "processed_data")
    log("repo {}".format(repo))

    R = load_runs(repo, args.data_root, args.date)
    D = reduce_runs(R, args.seed)
    log("writing to {}".format(out_dir))
    write_outputs(D, out_dir, args, repo, R["src"])


if __name__ == "__main__":
    main()
