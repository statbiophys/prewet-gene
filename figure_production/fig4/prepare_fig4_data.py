#!/usr/bin/env python3
"""Reduce the Fig. 4 solver output to the small tables fig4.ipynb reads.

    python figure_production/fig4/prepare_fig4_data.py

Each solver run holds the full concentration profile phi(z) at 512 far-field
values phi_inf (~19 MB). The figure uses much less:
  panels D-G  the surface value phi(z = 0) of the returned (lower-F) film, the
              metastable film's surface value where the run is bistable, the
              branch labels and convergence flags -- at every phi_inf;
  panel C     whole profiles, but only at the three phi_inf it marks (fractions
              SWEEP_FRACS of the binodal) of the three radius runs, for the
              decay lengths xi in its legend;
  panel B     the bulk parameters.

Reads   data/20261004/{small,medium,large}_R/demo_bvp_profiles.json
        data/20261007/run{1..5}/demo_bvp_profiles.json      (run_fig4_solver.sh)
Writes  processed_data/fig4_surface.csv    run, phi_inf index, phi_inf, surface values, branch
        processed_data/fig4_profiles.csv   z, then phi(z) of each panel-C profile (one column each)
        processed_data/fig4_scalars.json   per-run parameters, SWEEP_FRACS and their indices
Values are written at full precision (%.17g); read them with
float_precision="round_trip".
"""
import json, os, sys

RUNS_R = ["20261004/small_R", "20261004/medium_R", "20261004/large_R"]
RUNS_H = ["20261007/run%d" % i for i in range(1, 6)]
SWEEP_FRACS = [0.305, 0.5, 0.9]        # panel C's phi_inf / binodal


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    out = os.path.join(os.path.dirname(here), "processed_data")
    D = {}
    for run in RUNS_R + RUNS_H:
        src = os.path.join(repo, "data", run, "demo_bvp_profiles.json")
        if not os.path.exists(src):
            sys.exit("missing %s -- run figure_production/fig4/run_fig4_solver.sh first" % src)
        D[run] = json.load(open(src))

    phi_inf = D[RUNS_R[2]]["phi_inf"]
    binodal = D[RUNS_R[2]]["binodal"]
    for run, d in D.items():                       # the notebook assumes one shared grid
        assert d["phi_inf"] == phi_inf and d["z_solver"] == D[RUNS_R[0]]["z_solver"], run
    # nearest grid point to each fraction, as the notebook picks it
    inds = [min(range(len(phi_inf)), key=lambda i: abs(phi_inf[i] / binodal - t)) for t in SWEEP_FRACS]

    g = lambda x: "%.17g" % x
    with open(os.path.join(out, "fig4_surface.csv"), "w") as f:
        f.write("run,i,phi_inf,phi_surf,phi_surf_other,branch,success\n")
        for run, d in D.items():
            for i, p in enumerate(d["phi_inf"]):
                other = d["phi_other"][i]
                f.write(",".join([run, str(i), g(p), g(d["phi_solver"][i][0]),
                                  g(other[0]) if other else "nan", d["branch"][i],
                                  str(int(d["success"][i]))]) + "\n")

    cols = [(run, i) for run in RUNS_R for i in inds]
    z = D[RUNS_R[0]]["z_solver"]                   # phi_solver lives on z_solver
    with open(os.path.join(out, "fig4_profiles.csv"), "w") as f:
        f.write(",".join(["z"] + ["%s:%d" % c for c in cols]) + "\n")
        for k in range(len(z)):
            f.write(",".join([g(z[k])] + [g(D[run]["phi_solver"][i][k]) for run, i in cols]) + "\n")

    scal = {"sweep_fracs": SWEEP_FRACS, "sweep_inds": inds,
            "runs": {run: {"R": d["R"], "omega_b": d["omega_b"], "chi": d["chi"],
                           "binodal": d["binodal"], "params": {"surface": d["params"]["surface"],
                                                               "bvp": {"bc_reg": d["params"]["bvp"]["bc_reg"]}}}
                     for run, d in D.items()}}
    json.dump(scal, open(os.path.join(out, "fig4_scalars.json"), "w"), indent=1)
    for n in ("fig4_surface.csv", "fig4_profiles.csv", "fig4_scalars.json"):
        print("  %-20s %6.1f KB" % (n, os.path.getsize(os.path.join(out, n)) / 1e3))


if __name__ == "__main__":
    main()
