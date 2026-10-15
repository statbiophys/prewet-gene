"""Read the lattice Monte Carlo output of run_lattice_sweep.sh (Figs. 1-3).

A sweep folder holds output_cpu1.json ... output_cpu<nruns>.json, one run per
file; file k stores its run under keys "k" -> "1" (sim/sweep.jl's shard and
beta indices).
"""
import json
import os

import numpy as np


def read_run(sweep_dir, k):
    """The record of run k (1-based) of a sweep: a dict of observables."""
    with open(os.path.join(sweep_dir, "output_cpu{}.json".format(k))) as f:
        return json.load(f)[str(k)]["1"]


def load_sweeps(sweep_dirs, nruns, keys):
    """Observables of runs 1..nruns of each sweep folder.

    Returns {key: arr} with arr[i][k] the 1-D array of run k+1 in sweep_dirs[i].
    """
    out = {key: [] for key in keys}
    for sweep_dir in sweep_dirs:
        runs = [read_run(sweep_dir, k) for k in range(1, nruns + 1)]
        for key in keys:
            out[key].append([np.array(r.get(key, [])) for r in runs])
    return out
