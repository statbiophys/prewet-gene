# Parameters of the stages in this folder: each one includes this file and reads PARAMS.

using Droplet: Parameters
import JSON

# --- Output paths ------------------------------------------------------------
# FIG6_ROOT is the dated data folder, set by figure_production/fig6/run_fig6.sh
const FIG6_ROOT = get(ENV, "FIG6_ROOT", joinpath("data", "fig6"))

# --- BVP solver: two-sphere axisymmetric BVP grid, Newton solve, branch label -
Base.@kwdef struct BVPParams
    d_min_factor::Float64 = 0.0      # near-contact end of the d-sweep
    d_max_factor::Float64 = 0.5
    n_d::Int              = 17       # = ncpus - 1
    d_ref_factor::Float64 = 2.0      # reference gap (spheres far apart): U = F(d) - F_ref
    Nrho::Int             = 200
    dz_target::Float64    = 0.03*20
    rho_max::Float64      = 18.0*20
    z_buf::Float64        = 3.0*20
    f_tol::Float64        = 1e-6
    maxiter::Int          = 200
    bc_reg::Float64       = 0.00
    arc_dlambda0::Float64 = 0.0      # phi_inf progress on the first step; 0 => range/50
    arc_max_steps::Int    = 400      # continuation-step budget per gap
    bridge_frac::Float64  = 0.5
    # --- optional warm start for the bridged branch (sweep B) ----------------
    seed_path::String     = joinpath(FIG6_ROOT, "fig6_sweep")
    seed_save::Bool       = true
end

# --- Run mode ----------------------------------------------------------------
const RUN_MODE = get(ENV, "RUN_MODE", "production")
RUN_MODE in ("production", "smoke") || error("RUN_MODE=$RUN_MODE: use production | smoke")
const SMOKE    = RUN_MODE == "smoke"
const OUT_ROOT = SMOKE ? joinpath(joinpath(FIG6_ROOT, "fig6_s300", "run4"), "smoke") : joinpath(FIG6_ROOT, "fig6_s300", "run4")

function _select_s(s_values)
    haskey(ENV, "ROUSE_S") || return s_values[1]
    s = parse(Float64, ENV["ROUSE_S"])
    s in s_values || error("ROUSE_S=$s is not in PARAMS.rouse.s_values = $s_values")
    return s
end

# --- Rouse chain + thermal environment ---------------------------------------
Base.@kwdef struct RouseParams
    s_values::Vector{Float64} = SMOKE ? [300] : [300]
    s::Float64        = _select_s(s_values)   # beads between the two probe monomers (beadSep); derived, do not set
    a::Float64        = 20     # Kuhn / bond length (lattice sites) 20 b = 6 nm, 100 b = 30 nm
    kBT::Float64      = 1.0
    friction::Float64 = 1.3477088948787062e-08   # s / sites^2
end
D_eff(r::RouseParams) = 2 * r.kBT / r.friction

# --- FPT solver (downstream; carried here so PARAMS is the single source) -----
Base.@kwdef struct FPTParams
    potential::Symbol       = :bvp     # :bvp (nonlinear BVP U_full) or :linear (closed-form well)
    Ngrid::Int              = 1500     # grid points on [R, x_ridge] (escape / return problems)
    Nlocal::Int             = 100_000  # dense grid for the local-concentration integral
    ridge_scan_max::Float64 = 200     # x-range scanned to locate the barrier ridge
    ridge_scan_n::Int       = 8000
    plateau_tol::Float64    = 1e-3     # |dU_att/dx| (kBT/site) at which the largest-phi well has flattened; x_div = 1.2 x there
end

# --- Swept axes --------------------------------------------------------------
Base.@kwdef struct SweepParams
    phi_inf_values::Vector{Float64} = [] # collect(range(0.006, stop=0.012, length=64))
    phi_frac_min::Float64           = 0.1
    phi_frac_max::Float64           = 0.95
    n_phi::Int                      = 32
    omega_b_values::Vector{Float64} = [-0.008]
end

# --- Run metadata ------------------------------------------------------------
Base.@kwdef mutable struct SimulationParams
    spec_path::Union{Nothing, String}   = ""
    bvp_dir::String                     = joinpath(FIG6_ROOT, "fig6_sweep")
    output_dir::Union{Nothing, String}  = OUT_ROOT
    ncpus::Int          = 0    # placeholder; set to sweep.jl's ncpus (ARGS[3])
    interp_file::String = "bvp_potential_interp.json"
    fisher_file::String = "fisher_rate_from_fpt_solver.json"
    one_sphere_file::String = "demo_bvp_profiles.json"
end

# --- Master container --------------------------------------------------------
Base.@kwdef struct FisherParams
    surface::Parameters       = Parameters(nb = 4000.0, chi_b = 0.5280059665030825, 
                                           omega_b = -0.008,
                                           kappa = 5.19, phi_inf = 0.09, L = 100.0,
                                           R = 130, eps = 1e-8,
                                           conserve_volume = false,
                                           parallel_plates = false)
    bvp::BVPParams            = BVPParams()
    rouse::RouseParams        = RouseParams()
    fpt::FPTParams            = FPTParams()
    sweep::SweepParams        = SweepParams()
    sim::SimulationParams     = SimulationParams(output_dir = joinpath(OUT_ROOT, "s$(Int(rouse.s))"))
end

function dilute_binodal(chi::Real; nb::Real = 1.0)
    g(phi) = log(phi / (1.0 - phi)) + nb * chi * (1.0 - 2.0 * phi)
    lo, hi = 1e-8, 0.5 - 1e-9
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0; lo = mid; else; hi = mid; end
    end
    return 0.5 * (lo + hi)
end

function _bisect(f, a, b; n = 80)
    fa = f(a)
    for _ in 1:n
        m = 0.5 * (a + b)
        (fa * f(m) <= 0) ? (b = m) : (a = m; fa = f(m))
    end
    return 0.5 * (a + b)
end

function binodal(chi::Real; nb::Real = 1.0)
    p = Parameters(nb = nb, chi_b = chi, omega_b = 0.0, kappa = 1.0,
                   phi_inf = 0.1, L = 1.0, R = Inf, eps = 1e-16)
    lo_sp, hi_sp = compute_spinodal(p)
    lo_sp === nothing && return (nothing, nothing)
    dilute(mu) = _bisect(x -> dfb_dphi(x, p) - mu, 1e-300, lo_sp)
    dense(mu)  = _bisect(x -> dfb_dphi(x, p) - mu, hi_sp, 1 - 1e-16)
    gap(mu)    = (fb(dense(mu), p) - mu * dense(mu)) -
                 (fb(dilute(mu), p) - mu * dilute(mu))
    mu = _bisect(gap, dfb_dphi(hi_sp, p), dfb_dphi(lo_sp, p))
    return dilute(mu), dense(mu)
end

function bridge_threshold(P::FisherParams)
    lo, hi = binodal(P.surface.chi_b; nb = P.surface.nb)
    return lo + P.bvp.bridge_frac * (hi - lo)
end

function phi_inf_grid(P::FisherParams)
    s = P.sweep
    isempty(s.phi_inf_values) || return copy(s.phi_inf_values)
    b, _ = binodal(P.surface.chi_b; nb = P.surface.nb)
    return b .* collect(range(s.phi_frac_min, s.phi_frac_max; length = s.n_phi))
end

# --- JSON serialization ------------------------------------------------------
_ser(x::Union{Real,AbstractString}) = x
_ser(x::Symbol)                     = String(x)
_ser(x::AbstractVector)             = [_ser(v) for v in x]
_ser(x)                             = Dict(String(f) => _ser(getfield(x, f))
                                           for f in fieldnames(typeof(x)))

to_dict(P::FisherParams) = _ser(P)

function save_params(P::FisherParams, path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, to_dict(P), 4)
    end
    return path
end

const PARAMS = FisherParams()

