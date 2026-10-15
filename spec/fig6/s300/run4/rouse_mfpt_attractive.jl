# Rouse chain whose two tracked beads feel the two-sphere pair potential: escape
# (contact -> dividing surface) and return times at each phi_inf. Parameters come
# from param.jl; the potential and the dividing surface from the interpolation and
# FPT stages.
#
#   ./runJL sim/sweep.jl <this folder>/rouse_mfpt_attractive.jl <ncpus> <output_dir>

using Droplet
using Statistics
using Printf
using CubicSplines: CubicSpline
import JSON

include(joinpath(@__DIR__, "param.jl"))

slist = []
checkpoint_path = ""

# ── Run-length / storage knobs ───────────────────────────────────────────────
t_max       = SMOKE ? 2.0 : 240.0  # minutes of wall clock per target! call
save_mode   = :sojourn_only        # :sojourn_only | :distance_only
sample_skip = 100                  # subsampling of a stored distance trace
n_phi_run   = 14                    # phi_inf values to run, spread over the stored sweep
seed0       = 202610144

# ── Inputs ───────────────────────────────────────────────────────────────────
interp_path = joinpath(PARAMS.sim.output_dir, PARAMS.sim.interp_file)
fisher_path = joinpath(PARAMS.sim.output_dir, PARAMS.sim.fisher_file)
isfile(interp_path) || error("missing $interp_path -- run bvp_potential_interp.jl first")
isfile(fisher_path) || error("missing $fisher_path -- run fisher_rate_from_fpt_solver.jl first")
_ip = JSON.parsefile(interp_path)
_fp = JSON.parsefile(fisher_path)

_f(x)    = x === nothing ? NaN : Float64(x)              # JSON null -> NaN
_farr(v) = Float64[_f(x) for x in v]

const X_GRID   = _farr(_ip["x_grid"])
const PHI_ALL  = _farr(_ip["phi_inf"])
const U_ADS    = [_farr(r) for r in (PARAMS.fpt.potential == :bvp ?
                                     _ip["U_full_interp"] : _ip["U_linear"])]

const X_DIV      = _f(_fp["x_div"])
const REC_PHI    = [_f(r["phi_inf"])     for r in _fp["per_phi"]]
const REC_ESC    = [_f(r["escape_mfpt"]) for r in _fp["per_phi"]]
const REC_RET    = [_f(r["return_mfpt"]) for r in _fp["per_phi"]]
const REC_RIDGE  = [_f(r["x_ridge"])     for r in _fp["per_phi"]]
const REC_BARR   = [_f(r["barrier"])     for r in _fp["per_phi"]]

# ── Chain constants derived from PARAMS ──────────────────────────────────────
const KBT      = PARAMS.rouse.kBT
const FRICTION = PARAMS.rouse.friction
const KUHN     = PARAMS.rouse.a
const BEADSEP  = Int(PARAMS.rouse.s)              # bonds between the tracked beads
const NBEADS   = BEADSEP + 1
const STIFF    = 3 * KBT / KUHN^2                 # kappa, as in _set_scaling_params!
const TAU_BOND = FRICTION / STIFF                 # bond relaxation time
const DT       = 0.01 * TAU_BOND / 16                  # => dtDimless = 0.01
const TAU_ROUSE = TAU_BOND * BEADSEP^2 / π^2      # subchain relaxation time
const R        = PARAMS.surface.R

const CONTACT_TOL = (X_DIV - R) / (PARAMS.fpt.Ngrid - 1)
const A_BOUND     = R + CONTACT_TOL
const B_UNBOUND   = X_DIV

# ── Sweep axes ───────────────────────────────────────────────────────────────
alphas  = REC_PHI
betas   = PARAMS.sweep.omega_b_values
nparams = length(alphas)

if length(ARGS) >= 3
    let nc = parse(Int, ARGS[3])
        nparams % nc == 0 || @warn """ncpus does not divide nparams: sim/sweep.jl \
            will drop alphas[$(nc * div(nparams, nc) + 1):$nparams]$(nc > nparams ? " -- i.e. run nothing at all" : "").  \
            Use ncpus in $(filter(n -> nparams % n == 0, 1:nparams)).""" nparams ncpus=nc
    end
end

blurb = "Rouse chain in the two-sphere pair potential: the escape (contact -> " *
        "dividing surface) and return (dividing surface -> contact) first-passage " *
        "times measured from the A/B sojourns of the full 3D chain, against the " *
        "same two MFPTs from the 1D backward-FPE reduction in fisher_rate_from_" *
        "fpt_solver.jl.  Same potential, same landmarks, same phi_inf: the ratio " *
        "is the error of the Markovian reduction of the end-to-end coordinate."

function target!(res, args...)
    slist, keya, alpha, beta, blurb = args
    t0 = time()

    krec = argmin(abs.(REC_PHI .- alpha))
    kphi = argmin(abs.(PHI_ALL .- alpha))
    @assert isapprox(REC_PHI[krec], alpha; rtol = 1e-9) "no solver record at phi_inf=$alpha"
    @assert isapprox(PHI_ALL[kphi], alpha; rtol = 1e-9) "no U_ads row at phi_inf=$alpha"

    p = Droplet.with(PARAMS.surface; phi_inf = alpha, omega_b = beta)

    Urow   = U_ADS[kphi]
    slopel = (Urow[2] - Urow[1]) / (X_GRID[2] - X_GRID[1])
    uspl   = CubicSpline(X_GRID, Urow, extrapl = [slopel,], extrapr = [0,])
    adsorption_potential(x) = uspl[x]

    h_fd = 1e-6
    attractive_force = x ->
        -(adsorption_potential(x + h_fd) - adsorption_potential(x - h_fd)) / (2 * h_fd)

    total_potential(x) = adsorption_potential(x) - 2 * log(x) +
                         3 * x^2 / (2 * BEADSEP * KUHN^2)

    seed = seed0 + 1000 * BEADSEP + keya
    chain(n) = RousePolymer(;
        boltzmannConst  = 1.0,
        temperature     = KBT,
        friction        = FRICTION,
        kuhnLength      = KUHN,
        r_min           = R,
        nbeads          = Int32(NBEADS),
        nsteps          = n,
        dt              = DT,
        trackedBead1    = 1,
        trackedBead2    = NBEADS,
        attractiveForce = attractive_force,
        saveMode        = save_mode,
        a               = A_BOUND,
        b               = B_UNBOUND,
        seed            = seed,
    )

    # ── Size the run to the wall-clock budget ──
    probe = chain(1_000); init!(probe); simulate!(probe)
    n_probe, t_probe = 100_000, 0.0
    while true
        probe.nsteps = n_probe
        t_probe = @elapsed simulate!(probe)
        t_probe >= max(2, 0.6 * t_max) && break
        n_probe = ceil(Int, n_probe * 1.5 * max(2, 0.6 * t_max) / max(t_probe, 1e-3))
    end
    sec_per_step = t_probe / n_probe
    nsteps = max(floor(Int, (60 * t_max - (time() - t0)) / sec_per_step), 2)

    r = chain(nsteps)
    init!(r)
    simulate!(r)

    esc_times = Float64.(length(r.A_intervals) > 1 ? r.A_intervals[2:end] : Int[]) .* DT
    ret_times = Float64.(length(r.B_intervals) > 1 ? r.B_intervals[2:end] : Int[]) .* DT
    _mean(v)   = isempty(v) ? NaN : mean(v)
    _median(v) = isempty(v) ? NaN : median(v)

    res["blurb"]          = blurb
    res["alpha"]          = alpha
    res["beta"]           = beta
    res["keya"]           = keya
    res["s"]              = PARAMS.rouse.s
    res["run_mode"]       = RUN_MODE
    res["params"]         = to_dict(PARAMS)          # full parameter set, for provenance
    res["surface_params"] = Dict("nb" => p.nb, "chi_b" => p.chi_b, "kappa" => p.kappa,
                                 "omega_b" => p.omega_b, "R" => p.R, "L" => p.L,
                                 "phi_inf" => p.phi_inf)
    res["rouse_params"]   = Dict("nbeads" => NBEADS, "bead_separation" => BEADSEP,
                                 "kuhn_length" => KUHN, "kBT" => KBT,
                                 "friction" => FRICTION, "stiffness" => STIFF,
                                 "D_bead" => KBT / FRICTION, "D_eff" => D_eff(PARAMS.rouse),
                                 "dt" => DT, "nsteps" => nsteps,
                                 "t_max_min" => t_max, "sec_per_step" => sec_per_step,
                                 "seed" => seed,
                                 "tau_bond" => TAU_BOND, "tau_rouse" => TAU_ROUSE,
                                 "t_total" => nsteps * DT)

    res["x_contact"]   = A_BOUND
    res["contact_tol"] = CONTACT_TOL
    res["x_div"]       = B_UNBOUND
    res["x_ridge"]     = REC_RIDGE[krec]
    res["barrier"]     = REC_BARR[krec]
    res["well_depth"]  = adsorption_potential(R)

    res["A_intervals"]  = r.A_intervals
    res["B_intervals"]  = r.B_intervals
    res["escape_times"] = esc_times
    res["return_times"] = ret_times

    res["mfpt"] = Dict(
        "rouse_escape"    => _mean(esc_times),   "rouse_return"    => _mean(ret_times),
        "rouse_escape_median" => _median(esc_times),
        "rouse_return_median" => _median(ret_times),
        "solver_escape"   => REC_ESC[krec],      "solver_return"   => REC_RET[krec],
        "ratio_escape"    => _mean(esc_times) / REC_ESC[krec],
        "ratio_return"    => _mean(ret_times) / REC_RET[krec])
    res["n_escape"]  = length(esc_times)
    res["n_return"]  = length(ret_times)
    res["t_in_well"] = sum(r.A_intervals; init = 0) * DT
    res["t_in_bulk"] = sum(r.B_intervals; init = 0) * DT

    res["pot_grid"]        = X_GRID[1:4:end]
    res["ads_potential"]   = adsorption_potential.(X_GRID[1:4:end])
    res["total_potential"] = total_potential.(X_GRID[1:4:end])

    res["distance_traj"]    = r.distanceTraj === nothing ? Float64[] :
                              r.distanceTraj[1:sample_skip:end]
    res["sample_skip"]      = sample_skip
    res["trace_dt"]         = DT * sample_skip
    res["bead_relax_time"]  = r.timeScale
    res["bead_relax_scale"] = r.posScale
    res["runtime_min"]      = (time() - t0) / 60.0

    return res
end
