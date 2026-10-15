# Parameters of the stages in this folder: each one includes this file and reads PARAMS.

using Droplet: Parameters
import JSON

# --- BVP solver: two-sphere axisymmetric BVP grid, Newton solve, branch label -
Base.@kwdef struct BVPParams
    d_min_factor::Float64 = 0.0      # near-contact end of the d-sweep
    d_max_factor::Float64 = 0.5
    n_d::Int              = 63       # = ncpus - 1 for a 32-cpu run
    d_ref_factor::Float64 = 8.0      # reference gap (spheres far apart): U = F(d) - F_ref
    Nrho::Int             = 300
    dz_target::Float64    = 0.03
    rho_max::Float64      = 15.0
    z_buf::Float64        = 3.0
    f_tol::Float64        = 1e-6
    maxiter::Int          = 2000
    bc_reg::Float64       = 0.00
    arc_dlambda0::Float64 = 0.0      # phi_inf progress on the first step; 0 => range/50
    arc_max_steps::Int    = 400      # continuation-step budget per gap
    bridge_phi::Float64   = 0.5
end

# --- Rouse chain + thermal environment ---------------------------------------
Base.@kwdef struct RouseParams
    s::Float64        = 100.0   # number of beads between the two probe monomers (beadSep)
    a::Float64        = 1.0     # Kuhn / bond length (kuhnLength)
    kBT::Float64      = 1.0
    friction::Float64 = 1.0
end
D_eff(r::RouseParams) = 2 * r.kBT / r.friction

# --- FPT solver (downstream; carried here so PARAMS is the single source) -----
Base.@kwdef struct FPTParams
    potential::Symbol       = :bvp     # :bvp (nonlinear BVP U_full) or :linear (closed-form well)
    Ngrid::Int              = 1500     # grid points on [R, x_ridge] (escape / return problems)
    Nlocal::Int             = 100_000  # dense grid for the local-concentration integral
    ridge_scan_max::Float64 = 25.0     # x-range scanned to locate the barrier ridge
    ridge_scan_n::Int       = 8000
end

# --- Swept axes --------------------------------------------------------------
Base.@kwdef struct SweepParams
    phi_inf_values::Vector{Float64} = [] # collect(range(0.006, stop=0.012, length=64))
    phi_frac_min::Float64           = 0.001
    phi_frac_max::Float64           = 0.999
    n_phi::Int                      = 512
    omega_b_values::Vector{Float64} = [-0.56147]
end

# --- Run metadata ------------------------------------------------------------
Base.@kwdef mutable struct SimulationParams
    spec_path::Union{Nothing, String}   = ""
    output_dir::Union{Nothing, String}  = "data/20261007/run3"
    ncpus::Int          = 0    # placeholder; set to sweep.jl's ncpus (ARGS[3])
    interp_file::String = "bvp_potential_interp.json"
    fisher_file::String = "fisher_rate_from_fpt_solver.json"
    one_sphere_file::String = "demo_bvp_profiles.json"
end

# --- Master container --------------------------------------------------------
Base.@kwdef struct FisherParams
    surface::Parameters       = Parameters(nb = 2230.0, chi_b = 0.544821635760637, 
                                           omega_b = -0.0365,
                                           kappa = 10.12228759025409, phi_inf = 0.09, 
                                           L = 200.0,
                                           R = 100, eps = 1e-8,
                                           conserve_volume = false,
                                           parallel_plates = false)
    bvp::BVPParams            = BVPParams()
    rouse::RouseParams        = RouseParams()
    fpt::FPTParams            = FPTParams()
    sweep::SweepParams        = SweepParams()
    sim::SimulationParams     = SimulationParams()
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
    return dilute(mu) #, dense(mu)
end

function phi_inf_grid(P::FisherParams)
    s = P.sweep
    isempty(s.phi_inf_values) || return copy(s.phi_inf_values)
    b = binodal(P.surface.chi_b; nb = P.surface.nb)
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

