# Two-sphere BVP sweep over the gap d, by pseudo-arclength continuation in phi_inf:
# one record per gap, plus one at the reference gap. All parameters come from param.jl.
#
#   ./runJL sim/sweep.jl <this folder>/fisher_two_spheres.jl <ncpus> <output_dir>

using Droplet

include(joinpath(@__DIR__, "param.jl"))

slist = []
checkpoint_path = ""

const BINODAL, PHI_DENSE = binodal(PARAMS.surface.chi_b; nb = PARAMS.surface.nb)
const BRIDGE_PHI = bridge_threshold(PARAMS)

const _R  = PARAMS.surface.R
const _DZ = PARAMS.bvp.dz_target
const Z_BUF_GRID = ceil(Int, (_R + PARAMS.bvp.z_buf) / _DZ) * _DZ - _R
_snap(d::Real)   = round(d / _DZ) * _DZ
_nz_for(d::Real) = Int(round((4.0 * _R + _snap(d) + 2.0 * Z_BUF_GRID) / _DZ)) + 1

const D_GRID = _snap.(collect(range(PARAMS.bvp.d_min_factor * _R,
                                    PARAMS.bvp.d_max_factor * _R;
                                    length = PARAMS.bvp.n_d)))
const D_REF  = _snap(PARAMS.bvp.d_ref_factor * _R)

const PHI_TARGETS = phi_inf_grid(PARAMS)

alphas  = vcat(D_GRID, D_REF)
betas   = PARAMS.sweep.omega_b_values
nparams = length(alphas)

blurb = "Two-sphere BVP by pseudo-arclength continuation in phi_inf at fixed " *
        "gap.  Sharded over d (alphas; last entry is the reference gap) x " *
        "omega_b (betas).  Each record sweeps the whole phi_inf grid along one " *
        "solution curve, through both limit points of the capillary-bridging " *
        "bifurcation, and reports every branch (thin / saddle / bridged) at " *
        "every phi_inf together with the fold locations.  U = F - F_ref is " *
        "formed downstream in bvp_potential_interp.jl, which sees all d."

function _dense_seed(geom, phi_dense; reach = 2.0)
    R = geom.p.R
    x = Float64[]
    @inbounds for j in 1:geom.Nz, i in 1:geom.Nrho
        geom.unk_mask[i, j] || continue
        rho = geom.rho_grid[i]; z = geom.z_grid[j]
        s = min(sqrt(rho^2 + (z - geom.z_c1)^2),
                sqrt(rho^2 + (z - geom.z_c2)^2)) - R
        push!(x, s < reach ? phi_dense : geom.p.phi_inf)
    end
    return clamp.(x, geom.p.eps, 1.0 - geom.p.eps)
end

# --- Bridged-branch warm start from a previous solution ----------------------

function _fill_solid!(phi::Matrix{Float64}, fluid::AbstractMatrix{Bool};
                      passes::Int = 3)
    m = collect(fluid)
    Nrho, Nz = size(phi)
    for _ in 1:passes
        newly = falses(Nrho, Nz)
        @inbounds for j in 1:Nz, i in 1:Nrho
            m[i, j] && continue
            acc = 0.0; n = 0
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ii = i + di; jj = j + dj
                (1 <= ii <= Nrho && 1 <= jj <= Nz && m[ii, jj]) || continue
                acc += phi[ii, jj]; n += 1
            end
            n == 0 && continue
            phi[i, j] = acc / n; newly[i, j] = true
        end
        m .|= newly
    end
    return phi
end

function _bilin(phi, rho_grid, z_grid, rho, z)
    Nrho, Nz = size(phi)
    i  = clamp(searchsortedlast(rho_grid, rho), 1, Nrho - 1)
    j  = clamp(searchsortedlast(z_grid,   z),   1, Nz - 1)
    tr = clamp((rho - rho_grid[i]) / (rho_grid[i+1] - rho_grid[i]), 0.0, 1.0)
    tz = clamp((z   - z_grid[j])   / (z_grid[j+1]   - z_grid[j]),   0.0, 1.0)
    return (1 - tr) * (1 - tz) * phi[i, j]   + tr * (1 - tz) * phi[i+1, j] +
           (1 - tr) * tz       * phi[i, j+1] + tr * tz       * phi[i+1, j+1]
end

function _remap_seed(sd, geom)
    R = geom.p.R
    d_new = geom.d
    shift   = d_new - sd.d
    stretch = d_new > 0 ? sd.d / d_new : 1.0
    zmap(z) = z <= R ? z : z >= R + d_new ? z - shift : R + (z - R) * stretch
    x = Vector{Float64}(undef, count(geom.unk_mask))
    k = 0
    @inbounds for j in 1:geom.Nz, i in 1:geom.Nrho
        geom.unk_mask[i, j] || continue
        k += 1
        x[k] = _bilin(sd.phi, sd.rho_grid, sd.z_grid,
                      geom.rho_grid[i], zmap(geom.z_grid[j]))
    end
    return clamp.(x, geom.p.eps, 1.0 - geom.p.eps)
end

_seed_name(d, keyb) =
    "bridged_seed_d" * lpad(string(round(Float64(d); digits = 4)), 9, '0') *
    "_b$(keyb).json"

function _save_seed(dir, geom, phi, lam, omega_b, keyb, gap_center, F)
    mkpath(dir)
    path = joinpath(dir, _seed_name(geom.d, keyb))
    open(path, "w") do io
        JSON.print(io, Dict("kind"       => "two_sphere_bvp_seed",
                            "d"          => geom.d,
                            "R"          => geom.p.R,
                            "omega_b"    => omega_b,
                            "phi_inf"    => lam,
                            "gap_center" => gap_center,
                            "F"          => F,
                            "Nrho"       => geom.Nrho,
                            "Nz"         => geom.Nz,
                            "rho_grid"   => collect(geom.rho_grid),
                            "z_grid"     => collect(geom.z_grid),
                            "phi"        => vec(round.(_fill_solid!(copy(phi),
                                                                    geom.fluid_mask);
                                                       sigdigits = 8))))
    end
    return path
end

function _load_seed(path::AbstractString, d_new::Real, keyb::Int)
    isempty(path) && return nothing
    file = path
    if isdir(path)
        rx = r"^bridged_seed_d([0-9.]+)_b(\d+)\.json$"
        cands = [(parse(Float64, m.captures[1]), parse(Int, m.captures[2]), f)
                 for f in readdir(path; join = true)
                 for m in (match(rx, basename(f)),) if m !== nothing]
        if isempty(cands)
            @warn "seed_path holds no bridged_seed_*.json; sweep B falls back to \
                   the analytic dense seed" path
            return nothing
        end
        same  = filter(c -> c[2] == keyb, cands)
        pool  = isempty(same) ? cands : same
        below = filter(c -> c[1] <= d_new + 1e-9, pool)
        file  = isempty(below) ? pool[argmin([abs(c[1] - d_new) for c in pool])][3] :
                                 below[argmax([c[1] for c in below])][3]
    end
    if !isfile(file)
        @warn "seed file not found; sweep B falls back to the analytic dense \
               seed" file
        return nothing
    end
    j = JSON.parsefile(file)
    return (d        = Float64(j["d"]),
            R        = Float64(j["R"]),
            phi_inf  = Float64(j["phi_inf"]),
            rho_grid = Float64.(j["rho_grid"]),
            z_grid   = Float64.(j["z_grid"]),
            phi      = reshape(Float64.(j["phi"]), Int(j["Nrho"]), Int(j["Nz"])),
            path     = file)
end

const GAP_POLE = collect(range(0.0, 0.35; length = 8))

_phi_surf(geom, phi) =
    clamp(phi_on_sphere(phi, geom, geom.z_c1, GAP_POLE)[1], 0.0, 1.0)

_same_branch(a, b) = abs(a.F - b.F) < 1e-5 * max(1.0, abs(a.F)) &&
                     abs(a.gap_center - b.gap_center) < 1e-3

function target!(res, args...)
    slist, keya, alpha, beta, blurb = args

    t0 = time()

    bp      = PARAMS.bvp
    omega_b = beta
    d       = alpha
    is_ref  = keya == length(alphas)
    keyb    = something(findfirst(==(beta), betas), 1)   # names the seed file

    if keya == 1
        nc = PARAMS.sim.ncpus
        if nc > 0 && nparams % nc != 0
            lost = (nc * div(nparams, nc) + 1):nparams
            @warn """sim/sweep.jl will DROP the tail of `alphas`: set n_d = ncpus - 1 \
                     (nparams = n_d + 1 must be divisible by ncpus).""" nparams ncpus=nc dropped_alphas=collect(alphas[lost])
        end
    end

    p_lo = Droplet.with(PARAMS.surface;
                        phi_inf = first(PHI_TARGETS), omega_b = omega_b)
    R = p_lo.R
    phi_lo, phi_hi = extrema(PHI_TARGETS)
    phi_dense = PHI_DENSE

    geom = build_geometry(p_lo, d; rho_max = bp.rho_max, z_buf = Z_BUF_GRID,
                          Nrho = bp.Nrho, Nz = _nz_for(d), bc_reg = bp.bc_reg)
    frac = cell_fluid_fraction(geom)
    A_op = bp.bc_reg == 0.0 ? build_linear_operator(geom) : nothing
    work = (Matrix{Float64}(undef, geom.Nrho, geom.Nz),
            Matrix{Float64}(undef, geom.Nrho, geom.Nz))

    _F(phi) = excess_free_energy_two_spheres(
                  Dict{Symbol, Any}(:geom => geom, :phi => phi); frac = frac)

    found = [NamedTuple[] for _ in eachindex(PHI_TARGETS)]
    tgt_index = Dict(t => i for (i, t) in enumerate(PHI_TARGETS))
    n_polish_failed = Ref(0)
    seed_out = Ref{Union{Nothing, NamedTuple}}(nothing)

    function record!(x, lam, segment, stable)
        i = get(tgt_index, lam, 0)
        i == 0 && return
        xp, ok = polish_at_phi_inf(geom, x, lam; f_tol = bp.f_tol,
                                   maxiter = bp.maxiter, work = work, A = A_op)
        if !ok
            n_polish_failed[] += 1
            return
        end
        phi = scatter(xp, geom)          # geom.p is at lam, so F uses the right reference
        hit = (F = _F(phi), gap_center = gap_center_phi(geom, phi),
               phi_surf = _phi_surf(geom, phi), stable = stable, segment = segment)
        any(h -> _same_branch(h, hit), found[i]) && return
        push!(found[i], hit)
        if bp.seed_save && hit.gap_center > BRIDGE_PHI &&
           (seed_out[] === nothing || lam > seed_out[].phi_inf ||
            (lam == seed_out[].phi_inf && hit.gap_center > seed_out[].gap_center))
            seed_out[] = (phi_inf = Float64(lam), phi = phi,
                          gap_center = hit.gap_center, F = hit.F)
        end
        return
    end

    tr_lam = Float64[]; tr_F = Float64[]; tr_s = Float64[]
    tr_gc  = Float64[]; tr_seg = Int[]
    function trace_point!(x, lam, s, segment, stable)
        phi = scatter(x, geom)
        push!(tr_lam, lam); push!(tr_s, s); push!(tr_seg, segment)
        push!(tr_gc, gap_center_phi(geom, phi)); push!(tr_F, _F(phi))
        return
    end

    # --- Sweep A: up from the dilute end, along the thin-film branch ---
    seed_lo = solve_two_spheres(p_lo, d; rho_max = bp.rho_max, z_buf = Z_BUF_GRID,
                                Nrho = bp.Nrho, Nz = _nz_for(d),
                                f_tol = bp.f_tol, maxiter = bp.maxiter,
                                omega_continuation = true, bc_reg = bp.bc_reg)
    trace_a = ArcTrace()
    if seed_lo[:success]
        trace_a = arclength_phi_inf(geom, seed_lo[:x], phi_lo, phi_hi;
                                    targets = PHI_TARGETS, on_hit = record!,
                                    on_point = trace_point!,
                                    f_tol = bp.f_tol, maxiter = bp.maxiter,
                                    max_steps = bp.arc_max_steps,
                                    dlambda0 = bp.arc_dlambda0 > 0 ? bp.arc_dlambda0 :
                                               (phi_hi - phi_lo) / 50)
    end

    # --- Sweep B: down from the saturated end, dense-seeded ---
    seed      = _load_seed(bp.seed_path, d, keyb)
    seed_from = seed === nothing ? "" : seed.path
    lam_b     = seed === nothing ? phi_hi : seed.phi_inf
    p_hi      = Droplet.with(PARAMS.surface; phi_inf = lam_b, omega_b = omega_b)
    geom.p    = p_hi
    _cold_seed() = _dense_seed(geom, phi_dense; reach = max(2.0, 0.6 * d))
    _solve_b(x0) = solve_two_spheres(p_hi, d; rho_max = bp.rho_max,
                                     z_buf = Z_BUF_GRID, Nrho = bp.Nrho,
                                     Nz = _nz_for(d), f_tol = bp.f_tol,
                                     maxiter = bp.maxiter, x0_init = x0,
                                     omega_continuation = false, bc_reg = bp.bc_reg)
    seed_hi = _solve_b(seed === nothing ? _cold_seed() : _remap_seed(seed, geom))
    if seed !== nothing && !seed_hi[:success]
        @warn "remapped seed did not converge; sweep B falls back to the \
               analytic dense seed" d seed_d=seed.d file=seed.path
        seed   = nothing
        lam_b  = phi_hi
        p_hi   = Droplet.with(PARAMS.surface; phi_inf = lam_b, omega_b = omega_b)
        geom.p = p_hi
        seed_hi = _solve_b(_cold_seed())
    end
    trace_b = ArcTrace()
    trace_c = ArcTrace()
    ran_c   = false
    if seed_hi[:success]
        lam_start = seed === nothing ? phi_hi * 1.30 : lam_b
        if lam_start > phi_lo
            trace_b = arclength_phi_inf(geom, seed_hi[:x], lam_start, phi_lo;
                                        targets = PHI_TARGETS, on_hit = record!,
                                        f_tol = bp.f_tol, maxiter = bp.maxiter,
                                        max_steps = bp.arc_max_steps,
                                        dlambda0 = bp.arc_dlambda0 > 0 ? bp.arc_dlambda0 :
                                                   (phi_hi - phi_lo) / 50)
        end
        if lam_start < phi_hi
            ran_c = true
            trace_c = arclength_phi_inf(geom, seed_hi[:x], lam_start, phi_hi;
                                        targets = PHI_TARGETS, on_hit = record!,
                                        f_tol = bp.f_tol, maxiter = bp.maxiter,
                                        max_steps = bp.arc_max_steps,
                                        dlambda0 = bp.arc_dlambda0 > 0 ? bp.arc_dlambda0 :
                                                   (phi_hi - phi_lo) / 50)
        end
    else
        @warn "sweep B seed failed: bridged branches not reachable from sweep A \
               will be MISSING from this record" d omega_b phi_hi phi_dense
    end

    # --- Equilibrium: lowest F among the STABLE branches at each phi_inf ---
    nphi = length(PHI_TARGETS)
    F_eq       = fill(NaN, nphi)
    phi_surf   = fill(NaN, nphi)
    branch_eq  = zeros(Int, nphi)
    bridged_eq = falses(nphi)
    n_branch   = zeros(Int, nphi)
    F_all      = [Float64[] for _ in 1:nphi]
    stable_all = [Bool[]    for _ in 1:nphi]
    gapc_all   = [Float64[] for _ in 1:nphi]
    surf_all   = [Float64[] for _ in 1:nphi]
    bridged_all = [Bool[]   for _ in 1:nphi]

    for i in 1:nphi
        hits = sort(found[i]; by = h -> h.gap_center)
        n_branch[i] = length(hits)
        for h in hits
            push!(F_all[i], h.F); push!(stable_all[i], h.stable)
            push!(gapc_all[i], h.gap_center); push!(surf_all[i], h.phi_surf)
            push!(bridged_all[i], h.gap_center > BRIDGE_PHI)
        end
        st = findall(h -> h.stable, hits)
        isempty(st) && continue
        k = st[argmin([hits[j].F for j in st])]
        F_eq[i]        = hits[k].F
        phi_surf[i]    = hits[k].phi_surf
        branch_eq[i]   = hits[k].segment + 1      # 1-based, 0 = none
        bridged_eq[i]  = hits[k].gap_center > BRIDGE_PHI
    end

    keep = max(1, cld(length(tr_lam), 400))
    idx  = 1:keep:length(tr_lam)

    ok = isfinite.(F_eq)

    res["blurb"]           = blurb
    res["d"]               = d
    res["is_ref"]          = is_ref
    res["phi_inf"]         = PHI_TARGETS
    res["omega_b"]         = omega_b
    res["R"]               = R
    res["chi_b"]           = p_lo.chi_b
    res["binodal"]         = BINODAL
    res["binodal_dense"]   = PHI_DENSE
    res["bridge_phi"]      = BRIDGE_PHI
    res["s"]               = PARAMS.rouse.s
    res["a"]               = PARAMS.rouse.a
    res["d_grid"]          = D_GRID
    res["d_ref"]           = D_REF
    res["F"]               = F_all
    res["stable"]          = stable_all
    res["gap_center"]      = gapc_all
    res["phi_surf_branch"] = surf_all
    res["bridged"]         = bridged_all
    res["F_eq"]            = F_eq
    res["phi_surf"]        = phi_surf
    res["branch_eq"]       = branch_eq
    res["bridged_eq"]      = bridged_eq
    res["n_branch"]        = n_branch
    traces = ran_c ? (trace_a, trace_b, trace_c) : (trace_a, trace_b)
    res["folds"]           = vcat((t.folds for t in traces)...)
    res["fold_sweep"]      = vcat((fill(i, length(t.folds))
                                   for (i, t) in enumerate(traces))...)
    res["trace"]           = Dict("s"          => tr_s[idx],
                                  "phi_inf"    => tr_lam[idx],
                                  "F"          => tr_F[idx],
                                  "gap_center" => tr_gc[idx],
                                  "segment"    => tr_seg[idx])
    res["arc_status"]      = [String(t.status) for t in traces]
    res["n_arc_steps"]     = [length(t.lambda) for t in traces]
    res["n_newton"]        = sum(t.n_newton for t in traces)
    res["n_polish_failed"] = n_polish_failed[]
    res["n_phi_failed"]    = nphi - count(ok)
    res["success"]         = all(ok)
    outdir = PARAMS.sim.output_dir
    seed_file = ""
    if bp.seed_save && seed_out[] !== nothing && outdir !== nothing && !isempty(outdir)
        so = seed_out[]
        seed_file = _save_seed(outdir, geom, so.phi, so.phi_inf, omega_b, keyb,
                               so.gap_center, so.F)
    end
    res["seed_from"]       = seed_from
    res["seed_file"]       = seed_file
    res["runtime_min"]     = (time() - t0) / 60.0
    res["params"]          = to_dict(PARAMS)   # full parameter set, for provenance
    return res
end

