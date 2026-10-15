#!/usr/bin/env python3
"""Reduce the inputs of Fig. 1 to small files in processed_data/.

    python figure_production/fig1/prepare_fig1_data.py [--data-root DIR] [--date YYYYMMDD]

Reads
  <data-root>/<date>/proximal_promoter_transc_d_50_c_<c>/output_cpu<k>.json
      the proximal-promoter lattice runs, for panel E (c = 0.006, run 3, frames 0-40000).
Writes
  processed_data/fig1_trace.csv     smoothed transcription, distance, one row per frame;
                                    time = frame * mc_samples_per_frame (fig1_scalars.json)
  processed_data/fig1_scalars.json  provenance and constants
"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.lattice_runs import read_run    # figure_production/common/

EXP = "{date}/proximal_promoter_transc_d_50_c_{c:.3f}"
TRACE = (0.006, 3)
START, WIDTH = 0, 40000          # frames shown
BOX, RNAP = 62, 0.02             # total volume fraction (c + 0.02) * 62^3
SMOOTH_MCS = 500                 # window = 500 * int(sampling multiplier) frames


def resolve_repo(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "Project.toml")) and os.path.isdir(os.path.join(d, "spec")):
            return d
        d = os.path.dirname(d)
    sys.exit("repo root not found above " + start)


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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = resolve_repo(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(repo, "data"),
                    help="directory holding the dated run folders")
    ap.add_argument("--date", default=None,
                    help="run-date folder under --data-root (default: the latest one holding "
                         "a lattice sweep)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "processed_data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.date = a.date or latest_lattice_date(a.data_root)
    print("lattice runs:", os.path.join(a.data_root, a.date))
    scal = {"data_root_layout": EXP}

    # ---- E: transcription and probe-enhancer distance -------------------------------
    c, k = TRACE
    r = read_run(os.path.join(a.data_root, EXP.format(date=a.date, c=c)), k)
    mult = 50000 / ((c + RNAP) * BOX**3)          # MC samples per frame
    act = np.asarray(r["transcription_activity"], float)[START:START + WIDTH]
    dist = np.asarray(r["pattern_distances"], float)[START:START + WIDTH]
    t = np.arange(START, START + WIDTH) * mult / 1e5
    window = SMOOTH_MCS * int(mult)
    smooth = uniform_filter1d(act, size=window)
    with open(os.path.join(a.out, "fig1_trace.csv"), "w") as f:
        f.write("activity_smoothed,distance\n")
        for row in zip(smooth, dist):
            f.write("%.4g,%.4g\n" % row)      # 4 digits: far below a printed line width
    scal["trace"] = {"c": c, "run": k, "start": START, "width": WIDTH,
                     "mc_samples_per_frame": mult, "smoothing_window_frames": window,
                     "smoothing_window_mc_samples": window * mult}
    print("trace  c = %.3f run %2d: %d frames, t to %.3f e5, smoothing %d frames (%.0f MC samples)"
          % (c, k, len(t), t[-1], window, window * mult))

    json.dump(scal, open(os.path.join(a.out, "fig1_scalars.json"), "w"), indent=1)
    print("wrote fig1_* in", os.path.relpath(a.out, repo))


if __name__ == "__main__":
    main()
