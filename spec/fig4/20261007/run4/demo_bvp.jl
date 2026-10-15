# Single-sphere BVP: the profile phi(z) at each phi_inf of the sweep, and its
# linear-response approximation; writes demo_bvp_profiles.json to
# PARAMS.sim.output_dir. All parameters come from param.jl.
#
#   julia --project=@. <this folder>/demo_bvp.jl

using Droplet
import JSON

include(joinpath(@__DIR__, "param.jl"))

# --- Config drawn from PARAMS -------------------------------------------------
p0     = PARAMS.surface                 # base surface adsorption model
h0     = p0.omega_b                     # adsorption field (fixed across the sweep)
R      = p0.R
plates = p0.parallel_plates             # geometry switch: two plates vs one sphere
L_iso  = p0.L                           # far-field cutoff (sphere) / plate separation
bc_reg = PARAMS.bvp.bc_reg              # Robin-BC regularization (as fisher_two_spheres)
f_tol  = PARAMS.bvp.f_tol

mu_infs = Vector{Float64}()
phi_infs = phi_inf_grid(PARAMS)         # swept signal axis, as in fisher_two_spheres

N_solve = 400                           # 1D solver grid
N_eval  = 500                           # profile-sampling resolution
z       = collect(range(0.0, L_iso; length = N_eval))   # shared x-axis
BINODAL = binodal(PARAMS.surface.chi_b; nb = PARAMS.surface.nb)

# --- Sweep --------------------------------------------------------------------
geom = plates ? "parallel-plate (separation L = $L_iso)" : "isolated-sphere"
println("Solving $geom BVP over $(length(phi_infs)) phi_inf values ...")
phi_profiles_full   = Vector{Vector{Float64}}()
phi_profiles_other  = Vector{Vector{Float64}}()
phi_profiles_linear = Vector{Vector{Float64}}()
success             = Bool[]
z_solver    = Float64[]
phi_solver  = Vector{Vector{Float64}}()
dphi_solver = Vector{Vector{Float64}}()
p_solver    = Any[]
branch      = String[]
bistable    = Bool[]
F           = Float64[]
F_thin      = Float64[]
F_thick     = Float64[]
phi_other   = Vector{Vector{Float64}}()

for phi in phi_infs
    p   = Droplet.with(p0; phi_inf = phi)             # omega_b (=h0) preserved
    sol = plates ? solve_parallel_plates(p; N = N_solve, bc_reg = bc_reg, f_tol = f_tol) :
                   solve_isolated_sphere(p; L_iso = L_iso, N = N_solve, bc_reg = bc_reg, 
                                         f_tol = f_tol, track_branches = true)
    sol.success || @warn "BVP did not converge" phi_inf = phi

    full     = sol(z)                                 # phi(z) on the eval grid
    psi, _, _ = plates ? psi_linear_plates(z, p) : psi_linear(z, p)
    lin      = (plates ? phi : full[end]) .+ psi

    push!(mu_infs, compute_mu_inf(p))
    push!(phi_profiles_full,   full)
    push!(phi_profiles_linear, lin)
    push!(success, sol.success)

    isempty(z_solver) && append!(z_solver, sol.z)
    push!(phi_solver,  sol.phi)
    push!(dphi_solver, sol.dphi)
    push!(p_solver,    _ser(sol.p))                   # Parameters, field by field
    if !plates
        push!(branch,    String(sol.branch))
        push!(bistable,  sol.bistable)
        push!(F,         sol.F)
        push!(F_thin,    sol.F_thin)
        push!(F_thick,   sol.F_thick)
        push!(phi_other, sol.phi_other)
        dz_s = sol.z[2] - sol.z[1]
        push!(phi_profiles_other,
              sol.bistable ? [Droplet._linterp(sol.z, sol.phi_other,
                                               clamp(x, sol.z[1], sol.z[end]),
                                               dz_s, sol.z[1]) for x in z] :
                             fill(NaN, N_eval))
    end
end

println("  phi(0) = $(round(phi_profiles_full[1][1];  digits = 4))  (surface composition)")
if plates
    mid = phi_profiles_full[1][cld(N_eval, 2)]
    println("  phi(L/2) = $(round(mid; digits = 4))  (mid-gap; phi_inf = $(phi_infs[1]))")
else
    println("  phi(L) = $(round(phi_profiles_full[1][end]; digits = 4))  (≈ phi_inf = $(phi_infs[1]))")
end
println("  converged: $(count(success))/$(length(success))")
plates || println("  bistable: $(count(bistable))/$(length(bistable))  " *
                  "(thick film returned at $(count(==("thick"), branch)))")

# --- Write beside the pipeline output -----------------------------------------
out = Dict{String,Any}(
    "params"              => to_dict(PARAMS),
    "phi_inf"             => phi_infs,
    "mu_inf"              => mu_infs,
    "omega_b"             => h0,
    "R"                   => R,
    "binodal"             => BINODAL,
    "chi"                 => p0.chi_b,
    "L"                   => L_iso,                  # far-field cutoff / plate separation
    "parallel_plates"     => plates,
    "z_full"              => z,
    "z_linear"            => z,
    "phi_profiles_full"   => phi_profiles_full,      # per (phi_inf, z)
    "phi_profiles_linear" => phi_profiles_linear,    # per (phi_inf, z)
    "success"             => success,
    "z_solver"            => z_solver,               # solver grid of the raw fields
    "phi_solver"          => phi_solver,             # per (phi_inf, z_solver)
    "dphi_solver"         => dphi_solver,            # per (phi_inf, z_solver)
    "p_solver"            => p_solver,               # per phi_inf
)
plates || merge!(out, Dict{String,Any}(
    "branch"              => branch,
    "bistable"            => bistable,
    "F"                   => F,
    "F_thin"              => F_thin,
    "F_thick"             => F_thick,
    "phi_other"           => phi_other,              # per phi_inf, on z_solver
    "phi_profiles_other"  => phi_profiles_other,     # per (phi_inf, z)
))

data_dir = PARAMS.sim.output_dir
mkpath(data_dir)
outpath = joinpath(data_dir, PARAMS.sim.one_sphere_file)
open(outpath, "w") do io
    JSON.print(io, out)
end
println("saved -> $outpath")
