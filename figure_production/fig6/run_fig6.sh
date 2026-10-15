#!/usr/bin/env bash
#
# Fig. 6: the two-sphere and Rouse simulations behind processed_data/fig6_*.
# Only needed to regenerate the data; the figure itself draws from processed_data/.
#
# Run from the repo root, one or more stages, in this order:
#     bash figure_production/fig6/run_fig6.sh bvp        # two-sphere BVP sweep (warm-start passes)
#     bash figure_production/fig6/run_fig6.sh sweep      # 1D FPT solver + Rouse run, 30 concentrations
#     bash figure_production/fig6/run_fig6.sh potential  # U_attr(d) on the s = 110 grid (panel B)
#     bash figure_production/fig6/run_fig6.sh s300       # 4 independent runs at s = 300 (C, D, E)
#     bash figure_production/fig6/run_fig6.sh s_scan     # Fisher rate vs s, low and high c (H, I)
#     bash figure_production/fig6/run_fig6.sh all
#
# The specs are in spec/fig6/. The Rouse runs are long -- about a day of wall
# clock per concentration for `sweep`, 4 h (s300) and 12 h (s_scan) per process --
# and launch many processes at once: run them on a cluster node, with nohup.
#
# Environment:
#   DATE              output under data/$DATE (default: today); every stage of one
#                     regeneration must use the same DATE
#   RUN_MODE          production (default) | smoke: s300 and s_scan only -- a few
#                     minutes per chain, output under .../smoke, for checking the setup
#   BVP_NCPU          processes for the BVP sweep (default 18; must divide 18)
#   BVP_PASSES        BVP passes, each warm-started from the previous one (default 3)
#   SWEEP_NCPU        processes for the 30-concentration Rouse run (default 30; must divide 30)
#
# Output (gitignored), read by prepare_fig6_data.py:
#   data/$DATE/fig6_sweep/            BVP output_cpu*.json, bridged seeds, demo_bvp_profiles.json,
#                                     bvp_potential_interp.json, fisher_rate_from_fpt_solver.json, rouse/
#   data/$DATE/fig6_bvp_pass<k>/      the earlier BVP passes (seeds only; may be deleted)
#   data/$DATE/fig6_potential_s110/   bvp_potential_interp.json
#   data/$DATE/fig6_s300/run{1..4}/s300/
#   data/$DATE/fig6_s_scan/run{2,3,4}/{low_c,high_c}/s{50..600}/

set -euo pipefail

if [[ ! -f Project.toml || ! -f runJL ]]; then
    echo "ERROR: run this from the repo root (the directory holding Project.toml)." >&2
    exit 1
fi
[[ $# -ge 1 ]] || { sed -n '2,40p' "$0"; exit 1; }
command -v julia >/dev/null || { echo "ERROR: julia not found -- see figure_production/README.md, Fig. 4 step 1." >&2; exit 1; }

DATE=${DATE:-$(date +%Y%m%d)}
export FIG6_ROOT=data/$DATE
export RUN_MODE=${RUN_MODE:-production}
export OPENBLAS_NUM_THREADS=1       # concurrent processes must not each grab every core
BVP_NCPU=${BVP_NCPU:-18}
BVP_PASSES=${BVP_PASSES:-3}
SWEEP_NCPU=${SWEEP_NCPU:-30}
SPEC=spec/fig6
SWEEP=$FIG6_ROOT/fig6_sweep
(( 18 % BVP_NCPU == 0 )) || { echo "ERROR: BVP_NCPU must divide 18 (got $BVP_NCPU)" >&2; exit 1; }
(( 30 % SWEEP_NCPU == 0 )) || { echo "ERROR: SWEEP_NCPU must divide 30 (got $SWEEP_NCPU)" >&2; exit 1; }
need() { [[ -e "$1" ]] || { echo "ERROR: $1 missing -- run the '$2' stage first (same DATE)." >&2; exit 1; }; }
say() { echo "==> $(date '+%F %T')  $*"; }

echo "==> julia $(julia --version | awk '{print $3}'), instantiating the project; output under $FIG6_ROOT"
julia --project=@. -e 'using Pkg; Pkg.instantiate()'

stage_bvp() {
    # Pseudo-arclength BVP sweep over the gap d (18 gaps incl. the reference). The
    # bridged branch converges at large d only from a warm start, so the sweep is
    # repeated, each pass seeded from the bridged solutions the previous one saved.
    local seed="" out k
    for (( k = 1; k <= BVP_PASSES; k++ )); do
        if (( k == BVP_PASSES )); then out=$SWEEP; else out=$FIG6_ROOT/fig6_bvp_pass$k; fi
        say "BVP pass $k/$BVP_PASSES on $BVP_NCPU CPUs -> $out (seed: ${seed:-cold})"
        FIG6_SEED=$seed bash runJL sim/sweep.jl $SPEC/sweep/fisher_two_spheres.jl "$BVP_NCPU" "$out"
        seed=$out
    done
    say "single sphere (demo_bvp) -> $SWEEP/demo_bvp_profiles.json"
    julia --project=@. $SPEC/sweep/demo_bvp.jl
}

stage_sweep() {
    need "$SWEEP/demo_bvp_profiles.json" bvp
    say "interpolated potentials, FPT solver -> $SWEEP"
    julia --project=@. $SPEC/sweep/bvp_potential_interp.jl
    julia --project=@. $SPEC/sweep/fisher_rate_from_fpt_solver.jl
    say "Rouse run, 30 concentrations on $SWEEP_NCPU CPUs -> $SWEEP/rouse"
    bash runJL sim/sweep.jl $SPEC/sweep/rouse_mfpt_attractive.jl "$SWEEP_NCPU" "$SWEEP/rouse"
}

stage_potential() {
    # U_attr(d) does not depend on s, but its x-grid does (it runs to 4 a sqrt(s)), and
    # panel B is drawn on the s = 110 grid. The BVP sweep is s-independent (gaps in
    # units of R), so its records are copied rather than recomputed.
    need "$SWEEP/output_cpu1.json" bvp
    local out=$FIG6_ROOT/fig6_potential_s110
    mkdir -p "$out"
    cp "$SWEEP"/output_cpu*.json "$out"/
    say "U_attr on the s = 110 grid -> $out"
    julia --project=@. $SPEC/potential_s110/bvp_potential_interp.jl
}

# For every s in the spec's s_values, the chain
# interp -> FPT solver -> Rouse, all chains at once, then wait.
run_chains() {
    local here=$1 ncpu=$2 root svals s
    read -r root svals <<< "$(env -u ROUSE_S julia --project=@. -e "include(\"$here/param.jl\");
        print(OUT_ROOT, \" \", join(Int.(PARAMS.rouse.s_values), \" \"))")"
    [[ -n "${svals:-}" ]] || { echo "ERROR: could not read s_values from $here/param.jl" >&2; exit 1; }
    say "$here ($RUN_MODE): s = [$svals] -> $root"
    for s in $svals; do
        mkdir -p "$root/s$s"
        ( export ROUSE_S=$s
          julia --project=@. "$here/bvp_potential_interp.jl" &&
          julia --project=@. "$here/fisher_rate_from_fpt_solver.jl" &&
          bash runJL sim/sweep.jl "$here/rouse_mfpt_attractive.jl" "$ncpu" "$root/s$s/rouse"
        ) > "$root/s$s/run.log" 2>&1 \
            && say "  s=$s done" || say "  s=$s FAILED -- see $root/s$s/run.log" &
    done
    wait
}

stage_s300() {
    need "$SWEEP/demo_bvp_profiles.json" bvp
    for r in 1 2 3 4; do run_chains $SPEC/s300/run$r 30; done      # 30 processes per run
}

stage_s_scan() {
    need "$SWEEP/demo_bvp_profiles.json" bvp
    for r in 2 3 4; do for c in low_c high_c; do
        run_chains $SPEC/s_scan/run$r/$c 2                          # 12 s x 2 processes
    done; done
}

for st in "$@"; do
    case $st in
        bvp) stage_bvp ;; sweep) stage_sweep ;; potential) stage_potential ;;
        s300) stage_s300 ;; s_scan) stage_s_scan ;;
        all) stage_bvp; stage_sweep; stage_potential; stage_s300; stage_s_scan ;;
        *) echo "ERROR: unknown stage '$st' (bvp | sweep | potential | s300 | s_scan | all)" >&2; exit 1 ;;
    esac
done
say "Done. Next: python figure_production/fig6/prepare_fig6_data.py"
