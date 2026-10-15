#!/usr/bin/env python3
"""Reduce the lattice runs to the table of Fig. 3, in processed_data/.

    python figure_production/fig3/prepare_fig3_data.py [--data-root DIR] [--date YYYYMMDD] [--out DIR]

Reads   <data-root>/<date>/proximal_promoter_transc_d_50_c_<c>/output_cpu<k>.json
Writes  fig3_fisher_input.csv, fig3_scalars.json
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
THRESH_CLOSED = 6.5          # pattern_distance < thresh <=> "closed"

# SPRT parameters: a 10% concentration difference decided with error 0.32, and
# K = log((1-eps)/eps).
EPSILON = 0.32
DELTA_C_OVER_C = 0.10


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reduce the raw runs to the table Fig. 3 needs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default=None)
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--date", default=None)
    p.add_argument("--out", default=None,
                   help="output directory (default: figure_production/processed_data)")
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
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_runs(repo, data_root, date):
    src = os.path.join(os.path.abspath(os.path.expanduser(data_root)), date)
    if not os.path.isdir(src):
        raise SystemExit("no such data directory: {}".format(src))


    log("reading {} concentrations x {} runs from {} ...".format(NCONC, NRUNS, src))
    t0 = time.time()
    runs = load_sweeps([os.path.join(src, EXP.format(c)) for c in CONCS], NRUNS,
                       ("off_times", "pattern_distances", "local_concentration"))
    off_times = runs["off_times"]
    log("read in {:.0f} s".format(time.time() - t0))

    # rescale the event times to MC sweeps
    params = np.array(CONCS)
    mult = 50000 / ((params + 0.02) * 62 ** 3)
    for i in range(NCONC):
        for k in range(NRUNS):
            off_times[i][k] = off_times[i][k] * mult[i]

    return dict(off_times=off_times, pattern_distances=runs["pattern_distances"],
                local_concentration=runs["local_concentration"], src=src)


def reduce_runs(R):
    """Conditional means and occupancies, and the RNAP kinetics."""
    dists, conc = R["pattern_distances"], R["local_concentration"]

    def split(cidx, ridx, closed):
        v = np.array(conc[cidx][ridx])
        d = np.array(dists[cidx][ridx])
        idx = np.where(d < THRESH_CLOSED) if closed else np.where(d >= THRESH_CLOSED)
        return idx, v[idx]

    phi_closed = np.zeros(NCONC)
    phi_open = np.zeros(NCONC)
    p_closed = np.zeros(NCONC)
    p_open = np.zeros(NCONC)

    for c in range(NCONC):
        pooled_c, pooled_o = [], []
        n_c = n_o = n_tot = 0
        for r in range(NRUNS):
            ic, vc = split(c, r, True)
            io, vo = split(c, r, False)
            pooled_c = np.concatenate((pooled_c, vc))
            pooled_o = np.concatenate((pooled_o, vo))
            n_c += len(ic[0])
            n_o += len(io[0])
            n_tot += len(conc[c][r])
        phi_closed[c] = np.mean(pooled_c)
        phi_open[c] = np.mean(pooled_o)
        p_closed[c] = n_c / n_tot
        p_open[c] = n_o / n_tot

    mean_toff_rnap = float(np.mean(
        [np.mean(np.concatenate(R["off_times"][k])) for k in range(NCONC)]))

    theta = 40 * 1e3 / (6e23 * (25 * 1e-9) ** 3)      # volume fraction -> uM
    params = np.array(CONCS)
    df = pd.DataFrame({
        "c_phi": params, "c_uM": params * theta,
        "phi_local_closed": phi_closed, "phi_local_open": phi_open,
        "p_closed": p_closed, "p_open": p_open,
    })
    return df, theta, mean_toff_rnap


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
    os.makedirs(out_dir, exist_ok=True)

    R = load_runs(repo, args.data_root, args.date)
    fisher_df, theta, mean_toff = reduce_runs(R)

    # %.17g round-trips float64 exactly. This matters: the LOOCV behind panel B
    # is ill-conditioned at the small end of its lambda grid, and a rounded
    # input selects a different smoothing weight.
    fisher_df.to_csv(os.path.join(out_dir, "fig3_fisher_input.csv"),
                     index=False, float_format="%.17g")

    K = float(np.log((1 - EPSILON) / EPSILON))
    scalars = dict(
        theta=theta, theta_units="uM per unit volume fraction",
        mean_toff_rnap=mean_toff, K_D_bands=2.3e-3, K_D_curves=2.33e-3,
        phi_R=0.02, threshold_closed=THRESH_CLOSED,
        epsilon=EPSILON, K=K, delta_c_over_c=DELTA_C_OVER_C,
        sprt_note=("R(t)=log P(O(t)|c)/P(O(t)|c+dc) is a "
                   "drift-diffusion process with drift V and EQUAL diffusion "
                   "constant V; V=(1/2)(dc/c)^2*Fdot(c) and "
                   "<T_dec>=(K/V)tanh(K/2) with K=log((1-eps)/eps)."),
        n_conc=NCONC, n_runs=NRUNS,
        source_data=R["src"], source_date_folder=args.date,
        repo_commit=git_commit(repo), generated_by=os.path.basename(__file__),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with open(os.path.join(out_dir, "fig3_scalars.json"), "w") as f:
        json.dump(scalars, f, indent=2)
        f.write("\n")

    log("K = log((1-{})/{}) = {:.4f}".format(EPSILON, EPSILON, K))
    total = 0
    for name in ("fig3_fisher_input.csv", "fig3_scalars.json"):
        n = os.path.getsize(os.path.join(out_dir, name))
        total += n
        print("  {:28s} {:>8.1f} KB".format(name, n / 1024))
    print("  {:28s} {:>8.1f} KB".format("TOTAL", total / 1024))


if __name__ == "__main__":
    main()
