#!/usr/bin/env bash
#
# Lattice Monte Carlo behind Figs. 1-3 (spec/lattice/proximal_promoter_transcription.jl).
#
# Run from the repo root:
#     bash figure_production/run_lattice_sweep.sh C [ncpus]     # one concentration
#     bash figure_production/run_lattice_sweep.sh all [ncpus]   # all 16, one after the other
#
# C is the ligand volume fraction, one of 0.003, 0.004, ..., 0.018.
# Each concentration is 16 independent runs (NREPEATS) of T_MAX minutes of wall
# clock each -- 4320 min = 3 days. Every run is its own Julia
# process writing its own output_cpu<k>.json (the loaders expect one run per
# file); ncpus of them run at a time (default 16, i.e. all at once). The run
# length is wall-clock time, not a step count: do not run more processes than
# physical cores, or each run samples less. The whole data set is
# 16 x 16 = 256 three-day runs: a cluster job -- submit one concentration per
# 16-core node rather than using "all".
#
# Environment overrides:
#     T_MAX=10      minutes per run (default 4320; use a few minutes to test)
#     NREPEATS=16   runs per concentration
#     DATE=YYYYMMDD output folder name (default: today)
#
# Output: data/$DATE/proximal_promoter_transc_d_50_c_<C>/output_cpu<k>.json, the
# layout prepare_fig{1,2,3}_data.py read (--data-root data --date $DATE).

set -euo pipefail

if [[ ! -f Project.toml || ! -f runJL ]]; then
    echo "ERROR: run this from the repo root (the directory holding Project.toml)." >&2
    exit 1
fi
command -v julia >/dev/null || { echo "ERROR: julia not found -- see README.md, 'Julia'." >&2; exit 1; }
[[ $# -ge 1 ]] || { sed -n '4,8p' "$0" >&2; exit 1; }

NREPEATS=${NREPEATS:-16}
NCPUS=${2:-16}
T_MAX=${T_MAX:-4320}
DATE=${DATE:-$(date +%Y%m%d)}

if [[ $1 == all ]]; then
    CONCS=$(python3 -c 'print(" ".join("%.3f" % (0.003 + 0.001 * i) for i in range(16)))')
else
    CONCS=$(printf "%.3f" "$1")
fi

echo "==> julia $(julia --version | awk '{print $3}'), instantiating the project"
julia --project=@. -e 'using Pkg; Pkg.instantiate()'

for C in $CONCS; do
    OUT=data/$DATE/proximal_promoter_transc_d_50_c_$C
    echo "==> c = $C: $NREPEATS runs x $T_MAX min, $NCPUS at a time -> $OUT   (started $(date +%H:%M))"
    # sim/sweep.jl SPEC k N OUT runs the k-th of N shards; with N = NREPEATS each
    # shard is one run, saved as output_cpu<k>.json
    for (( k = 1; k <= NREPEATS; k++ )); do
        DROPLET_TF_CONC=$C DROPLET_T_MAX=$T_MAX DROPLET_NREPEATS=$NREPEATS \
            julia --project=@. sim/sweep.jl spec/lattice/proximal_promoter_transcription.jl \
            "$k" "$NREPEATS" "$OUT" &
        (( k % NCPUS == 0 )) && wait
    done
    wait
    n=$(ls "$OUT"/output_cpu*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "    $n output files (expected $NREPEATS)"
    [[ $n == "$NREPEATS" ]] || { echo "ERROR: missing output in $OUT" >&2; exit 1; }
done

echo
echo "Done ($(date +%H:%M)). Next: python figure_production/fig2/prepare_fig2_data.py --data-root data --date $DATE"
