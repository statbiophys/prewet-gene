#!/usr/bin/env python3
"""Reduce the Fig. 6 simulations to small files in processed_data/.

    python figure_production/fig6/prepare_fig6_data.py [--sweep DIR] [--potential DIR]
                                                       [--s300 DIR] [--s-scan DIR] [--out DIR]

Each option defaults to the newest matching folder written by fig6/run_fig6.sh:
data/<date>/fig6_sweep, fig6_potential_s110, fig6_s300, fig6_s_scan.

Writes (processed_data/):
  fig6_a.csv         A: probe volume fraction vs d, low and high c
  fig6_b.csv         B: attractive force vs d, low and high c
  fig6_vs_c.csv      C, D, E: P(closed), cycle times, Fisher rates vs c
  fig6_ef.csv        F, G: escape and return time PDFs
  fig6_vs_s.csv      H, I: Fisher rates vs s
  fig6_scalars.json  c_sat, s*, concentrations, settings
"""
import argparse, glob, json, os, sys

sys.dont_write_bytecode = True        # never write __pycache__ into common/
import numpy as np
from scipy.ndimage import gaussian_filter1d

if not hasattr(np, "trapezoid"):      # numpy < 2: common/rouse_fisher.py calls np.trapezoid
    np.trapezoid = np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))             # figure_production/
from common import rouse_fisher       # noqa: E402



def _latest(name):
    found = sorted(glob.glob(os.path.join(REPO, "data", "*", name)))
    return found[-1] if found else None


ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--sweep", default=_latest("fig6_sweep"),
                help="two-sphere sweep + Rouse run (default: latest data/*/fig6_sweep)")
ap.add_argument("--potential", default=_latest("fig6_potential_s110"),
                help="U_attr(d) run (default: latest data/*/fig6_potential_s110)")
ap.add_argument("--s300", default=_latest("fig6_s300"),
                help="folder holding run*/s300 (default: latest data/*/fig6_s300)")
ap.add_argument("--s-scan", dest="s_scan", default=_latest("fig6_s_scan"),
                help="folder holding run*/{low_c,high_c} (default: latest data/*/fig6_s_scan)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "processed_data"),
                help="where fig6_* are written (default: figure_production/processed_data)")
args = ap.parse_args()
for _k in ("sweep", "potential", "s300", "s_scan"):
    _v = getattr(args, _k)
    if not _v or not os.path.isdir(_v):
        sys.exit("--%s: no such run folder (%s); see figure_production/README.md" % (_k.replace("_", "-"), _v))
SWEEP, POTENTIAL, S300, S_SCAN = args.sweep, args.potential, args.s300, args.s_scan
OUT = args.out
os.makedirs(OUT, exist_ok=True)

NM_PER_SITE = 0.3                     # lattice spacing, nm
R_NM = 40.0                           # sphere radius in nm: d = x * 0.3 + 40

# free receptor
V_TF = 120                            # nm^3 per transcription factor (<-> nb = 4000)
A_REC = 3e-3                          # receptor radius, um
D_TF = 7.4                            # um^2/s
K_D_FRAC = 0.325                      # K_D / c_sat

# Rouse time scale: simulated seconds -> plotted seconds
COEF_EF = 50                          # F, G
COEF_CDH = 100                        # C, D, E, H, I


def jload(root, *parts):
    with open(os.path.join(root, *parts)) as f:
        return json.load(f)


def particles_per_um3(phi):
    return phi / V_TF * 1e9


def rouse_records(directory):
    """phi -> list of Rouse records (one per run file) under directory/output_cpu*.json."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "output_cpu*.json"))):
        with open(path) as f:
            for per_cpu in json.load(f).values():
                for record in per_cpu.values():
                    out.setdefault(record["alpha"], []).append(record)
    return out


def g(x):
    return "nan" if not np.isfinite(x) else "%.6g" % x      # plot data: 6 digits is plenty


def write_csv(name, columns):
    keys = list(columns)
    n = len(columns[keys[0]])
    with open(os.path.join(OUT, name), "w") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            f.write(",".join(v[i] if isinstance(v[i], str) else g(float(v[i])) for v in columns.values()) + "\n")


# ---------------------------------------------------------------------------------------
# single sphere, free receptor
# ---------------------------------------------------------------------------------------
single = jload(SWEEP, "demo_bvp_profiles.json")
phi_inf = np.array(single["phi_inf"])
# "phi_profiles_full" at the surface (column 0) is the only value used here
phi_full = np.array(single.get("phi_solver", single["phi_profiles_full"]))
binodal = float(np.atleast_1d(single["binodal"])[0])
kb = 4 * A_REC * D_TF
K_D = K_D_FRAC * particles_per_um3(binodal)
ku = kb * K_D
tau_free = 1 / (kb * particles_per_um3(phi_inf)) + 1 / ku
p_free = kb * particles_per_um3(phi_inf) / (kb * particles_per_um3(phi_inf) + ku)
fisher_rate_free = 1 / tau_free
phi_open = phi_full[:, 0]
tau_burst_open = 1 / (kb * particles_per_um3(phi_open)) + 1 / ku
fisher_rate_open = (np.gradient(phi_open, phi_inf) * phi_inf / phi_open) ** 2 / tau_burst_open
scal = {"c_sat_phi": binodal, "K_D_over_c_sat": K_D_FRAC, "coef_EF": COEF_EF, "coef_CDH": COEF_CDH}


# ---------------------------------------------------------------------------------------
# A, B
# ---------------------------------------------------------------------------------------
two = jload(SWEEP, "fisher_rate_from_fpt_solver.json")
x_ret = np.array(two["per_phi"][0]["x_grid_ret"])
gap_profiles = np.array([r["phi_surf"] for r in two["per_phi"]])
interp = jload(POTENTIAL, "bvp_potential_interp.json")
x_grid = np.array(interp["x_grid"])[:-1]
z = x_grid * NM_PER_SITE
ab_a = {"d_nm": x_ret * NM_PER_SITE + R_NM}
ab_b = {"d_nm": z + R_NM}
scal["A"], scal["B"] = {}, {}
# both panels coloured on viridis over the s110 run's c/c_sat range
scal["AB_colour_range"] = [min(interp["phi_inf"]) / binodal, max(interp["phi_inf"]) / binodal]
for tag, k in (("low", 1), ("high", -1)):
    ab_a["phi_local_" + tag] = gap_profiles[k]
    scal["A"][tag] = two["per_phi"][k]["phi_inf"] / binodal
    u = np.array(interp["U_full_interp"])[k][:-1]
    ab_b["force_" + tag] = -np.gradient(u, z)
    scal["B"][tag] = interp["phi_inf"][k] / binodal
write_csv("fig6_a.csv", ab_a)
write_csv("fig6_b.csv", ab_b)
print("A at c/c_sat = %(low).3f, %(high).3f" % scal["A"], "| B at c/c_sat = %(low).3f, %(high).3f" % scal["B"])


# ---------------------------------------------------------------------------------------
# F, G: escape and return time PDFs
# ---------------------------------------------------------------------------------------
MIN_EVENTS, N_GRID, SMOOTH_SIGMA = 50, 300, 20


def cdf_to_pdf(times):
    t = np.sort(np.asarray(times, float))
    t = t[t > 0]
    if t.size < 2:
        return None, None
    grid = np.logspace(np.log10(t[0]), np.log10(t[-1]), N_GRID)
    u = np.log10(grid)
    F = np.searchsorted(t, grid, side="right") / t.size
    F = gaussian_filter1d(F, sigma=SMOOTH_SIGMA, mode="nearest")
    return grid, np.clip(np.gradient(F, u) / (grid * np.log(10.0)), 0, None)


ef = rouse_records(os.path.join(SWEEP, "rouse"))
shown = [a for a in sorted(ef)
         if max((np.asarray(ef[a][0]["escape_times"]) > 0).sum(),
                (np.asarray(ef[a][0]["return_times"]) > 0).sum()) >= MIN_EVENTS]
rows = {"channel": [], "c_over_csat": [], "t": [], "pdf": []}
for a in shown:
    for channel in ("escape", "return"):
        t = np.asarray(ef[a][0][channel + "_times"]) * COEF_EF
        t = t[t > 0]
        if t.size < MIN_EVENTS:
            continue
        grid, pdf = cdf_to_pdf(t)
        keep = pdf > 0
        n = int(keep.sum())
        rows["channel"] += [channel] * n
        rows["c_over_csat"] += [a / binodal] * n
        rows["t"] += list(grid[keep])
        rows["pdf"] += list(pdf[keep])
write_csv("fig6_ef.csv", rows)
scal["EF"] = {"c_over_csat_min": min(shown) / binodal, "c_over_csat_max": max(shown) / binodal,
              "n_shown": len(shown), "n_total": len(ef)}
print("F, G: %d/%d concentrations" % (len(shown), len(ef)))


# ---------------------------------------------------------------------------------------
# C, D, E
# ---------------------------------------------------------------------------------------
settings = rouse_fisher.FisherSettings(
    time_rescale=COEF_CDH, n_histogram_bins=40, min_cycles_per_phi=300, n_folds=5,
    block_length=50, n_bootstrap=200, n_permutations=50, min_signal_to_noise=3, seed=20261014)
run_dirs = sorted(glob.glob(os.path.join(S300, "run*", "s300")))
runs_at_phi = {}
for rd in run_dirs:
    for a, recs in rouse_records(os.path.join(rd, "rouse")).items():
        runs_at_phi.setdefault(a, []).extend(recs)
phi_sim = np.array(sorted(runs_at_phi))
with open(os.path.join(run_dirs[0], "fisher_rate_from_fpt_solver.json")) as f:
    solver = json.load(f)
solver_at_phi = {r["phi_inf"]: r for r in solver["per_phi"]}
assert np.allclose(phi_sim, phi_inf[:phi_sim.size], rtol=1e-9)
mismatch = max(abs(solver_at_phi[p]["phi_surf"][-1] / phi_full[k, 0] - 1)
               for k, p in enumerate(sorted(solver_at_phi)))
assert mismatch < 1e-2, mismatch
print("%d runs at s = 300, %d concentrations; single sphere agrees with the solver to %.1f%%"
      % (len(run_dirs), phi_sim.size, 100 * mismatch))

STEP = 2
n_pts = phi_sim.size - STEP
phi_pts = phi_sim[:n_pts]
pairs = [dict(runs_base=runs_at_phi[phi_sim[k]], runs_plus=runs_at_phi[phi_sim[k + STEP]],
              solver_base=solver_at_phi[phi_sim[k]], solver_plus=solver_at_phi[phi_sim[k + STEP]],
              phi_base=phi_sim[k], phi_plus=phi_sim[k + STEP]) for k in range(n_pts)]
vs_c = rouse_fisher.analyse_sweep(phi_pts, pairs, settings, x_name="phi")
polymer = vs_c["escape"]

phi_closed = np.array([solver_at_phi[p]["phi_local"] for p in phi_pts])
dphi_closed = np.array([solver_at_phi[p]["dphi_local"] for p in phi_pts])
tau_burst_closed = 1 / (kb * particles_per_um3(phi_closed)) + 1 / ku
rate_closed = (dphi_closed * phi_pts / phi_closed) ** 2 / tau_burst_closed
p_closed = vs_c["p_closed"]
rate_open = fisher_rate_open[:n_pts]
rate_local = (1 - p_closed) * rate_open + p_closed * rate_closed
local_reps = ((1 - vs_c["p_closed_bootstrap"]) * rate_open[:, None]
              + vs_c["p_closed_bootstrap"] * rate_closed[:, None])
total_fit = polymer["fit"] + rate_local
total_reps = polymer["fit_replicates"] + local_reps.T
spread = np.std(np.log10(total_reps), axis=0, ddof=1)
write_csv("fig6_vs_c.csv", {
    "c_over_csat": phi_pts / binodal,
    "analysed": np.isfinite(vs_c["cycle_time"]).astype(int),
    "p_closed": p_closed, "p_closed_se": vs_c["p_closed_se"], "p_free": p_free[:n_pts],
    "cycle_time": vs_c["cycle_time"], "cycle_time_se": vs_c["cycle_time_se"], "tau_free": tau_free[:n_pts],
    "rate_free": fisher_rate_free[:n_pts], "rate_open": rate_open,
    "polymer_fit": polymer["fit"], "polymer_lo": polymer["band"][0], "polymer_hi": polymer["band"][1],
    "total_fit": total_fit, "total_lo": total_fit * 10 ** -spread, "total_hi": total_fit * 10 ** spread,
})


# ---------------------------------------------------------------------------------------
# H, I: Fisher rates vs s
# ---------------------------------------------------------------------------------------
root = S_SCAN
conds = sorted({os.path.basename(p) for p in glob.glob(root + "/run*/*") if os.path.isdir(p)})
runs_c = {c: sorted(glob.glob("%s/run*/%s" % (root, c))) for c in conds}
rouse_s = {}
for cond, cruns in runs_c.items():
    data_s, fisher_s, solver_s = {}, {}, {}
    for rdir in cruns:
        for sdir in sorted(glob.glob(rdir + "/s[0-9]*")):
            s = int(os.path.basename(sdir)[1:])
            if s > 600 or not glob.glob(sdir + "/rouse/output_cpu*.json"):
                continue
            for a, recs in rouse_records(sdir + "/rouse").items():
                data_s.setdefault(s, {}).setdefault(a, []).extend(recs)
            if s not in fisher_s:
                with open(sdir + "/fisher_rate_from_fpt_solver.json") as f:
                    fisher_s[s] = json.load(f)
                per_phi = {r["phi_inf"]: r for r in fisher_s[s]["per_phi"]}
                solver_s[s] = {a: per_phi[min(per_phi, key=lambda p: abs(p - a))] for a in data_s[s]}
    svals = np.array(sorted(data_s))
    rouse_s[cond] = dict(data=data_s, fisher=fisher_s, solver=solver_s, s=svals,
                         phi=np.array(sorted(data_s[svals[0]])))
last_fisher = rouse_s[conds[-1]]["fisher"]

settings_s = rouse_fisher.FisherSettings(
    time_rescale=COEF_CDH, n_histogram_bins=40, min_cycles_per_phi=300, n_folds=5,
    block_length=50, n_bootstrap=200, n_permutations=50, min_signal_to_noise=3, seed=20261013)
fpoly = {}
for cond, cd in rouse_s.items():                  # high_c, then low_c (fixes the seed stream)
    pb, pp = cd["phi"][0], cd["phi"][-1]
    pairs = [dict(runs_base=cd["data"][s][pb], runs_plus=cd["data"][s][pp],
                  solver_base=cd["solver"][s][pb], solver_plus=cd["solver"][s][pp],
                  phi_base=pb, phi_plus=pp) for s in cd["s"]]
    fpoly[cond] = rouse_fisher.analyse_sweep(cd["s"], pairs, settings_s, x_name="s")
    fpoly[cond].update(s=cd["s"], phi_base=pb)


def first_ridge(x, attraction, s, a):
    slope = np.gradient(attraction - 2 * np.log(x / (a * np.sqrt(s))) + 3 * x ** 2 / (2 * s * a ** 2), x)
    k = np.flatnonzero((slope[:-1] > 0) & (slope[1:] <= 0))
    return x[k[0]] if k.size else None


def ridge_onset(x, attraction, a, s_max, tol=0.01):
    lo, hi = 1.0, float(s_max)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if first_ridge(x, attraction, mid, a) is None:
            lo = mid
        else:
            hi = mid
    return hi


PHI_INDEX = {"low_c": 2, "high_c": 12}
s_rows = {k: [] for k in ("condition", "s", "rate_free", "rate_open", "polymer_fit", "polymer_lo",
                          "polymer_hi", "rate_local", "rate_local_se")}
scal["s_star"], scal["IJ_c_over_csat"] = {}, {}
for cond in ("low_c", "high_c"):
    pol, svals = fpoly[cond], fpoly[cond]["s"]
    i = PHI_INDEX[cond]
    assert np.isclose(phi_inf[i], pol["phi_base"], rtol=1e-9)
    pc = np.array([last_fisher[s]["per_phi"][i]["phi_local"] for s in svals])
    dpc = np.array([last_fisher[s]["per_phi"][i]["dphi_local"] for s in svals])
    rc = (dpc * phi_inf[i] / pc) ** 2 / (1 / (kb * particles_per_um3(pc)) + 1 / ku)
    ro = fisher_rate_open[i]
    rl = (1 - pol["p_closed"]) * ro + pol["p_closed"] * rc
    rl_se = np.abs(rc - ro) * pol["p_closed_se"]
    with open("%s/s%d/bvp_potential_interp.json" % (runs_c[cond][0], int(svals.min()))) as f:
        ip = json.load(f)
    row = int(np.argmin(np.abs(np.array(ip["phi_inf"]) - pol["phi_base"])))
    a_kuhn = ip["params"]["rouse"]["a"]
    att = np.array(ip["U_full_interp"][row], float)
    x = np.linspace(ip["R"], 4 * a_kuhn * np.sqrt(svals.max()), 40_000)
    att = np.interp(x, ip["x_grid"], att, right=att[-1])
    scal["s_star"][cond] = ridge_onset(x, att, a_kuhn, svals.max())
    scal["IJ_c_over_csat"][cond] = pol["phi_base"] / binodal
    for k, s in enumerate(svals):
        for key, v in (("condition", cond), ("s", s), ("rate_free", fisher_rate_free[i]), ("rate_open", ro),
                       ("polymer_fit", pol["escape"]["fit"][k]), ("polymer_lo", pol["escape"]["band"][0, k]),
                       ("polymer_hi", pol["escape"]["band"][1, k]), ("rate_local", rl[k]),
                       ("rate_local_se", rl_se[k])):
            s_rows[key].append(v)
    print("%s: c/c_sat = %.2f, s* = %.2f" % (cond, scal["IJ_c_over_csat"][cond], scal["s_star"][cond]))
    print("     s  p_closed  F_local  F_poly  F_total")
    for k, s in enumerate(svals):
        print("  %4d   %6.3f   %6.2f  %6.1f  %6.1f" % (s, pol["p_closed"][k], rl[k], pol["escape"]["rate"][k],
                                                    pol["escape"]["rate"][k] + rl[k]))
write_csv("fig6_vs_s.csv", s_rows)

scal["sources"] = {"A": os.path.join(SWEEP, "fisher_rate_from_fpt_solver.json"),
                   "B": os.path.join(POTENTIAL, "bvp_potential_interp.json"),
                   "EF": os.path.join(SWEEP, "rouse"), "CDH": os.path.join(S300, "run*", "s300"),
                   "IJ": os.path.join(S_SCAN, "run*", "{low_c,high_c}"),
                   "single_sphere": os.path.join(SWEEP, "demo_bvp_profiles.json"),
                   }
json.dump(scal, open(os.path.join(OUT, "fig6_scalars.json"), "w"), indent=1)
print("wrote fig6_* in", os.path.relpath(OUT, REPO))
