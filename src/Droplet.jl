module Droplet

    using Random
    using StatsBase
    using Serialization
    using LinearAlgebra
    using JSON
    using OrderedCollections
    using SparseArrays
    using Printf
    using NonlinearSolve
    using NonlinearSolve.ADTypes: AutoFiniteDiff
    using NonlinearSolve.LinearSolve: KrylovJL_GMRES
    using NonlinearSolve.SciMLBase

    # Lattice Monte Carlo (Figs. 1-3): species and lattice, interactions and
    # energies, Metropolis moves, initialisation, checkpoints, measurements.
    include("types.jl")
    include("interactions.jl")
    include("hamiltonian.jl")
    include("metropolis.jl")
    include("move.jl")
    include("port.jl")
    include("init.jl")
    include("utils.jl")
    include("stats.jl")

    # Rouse chain with a pair force between two tracked beads (Fig. 5).
    include("rouse.jl")

    # Flory-Huggins thermodynamics and the 1D isolated-sphere BVP for surface
    # condensation (Fig. 4); its linear response gives the pair potential of
    # Fig. 5. Independent of the Metropolis core.
    include("bvp_solver.jl")

    # Two spheres (Fig. 6): the 2D axisymmetric BVP, its pseudo-arclength
    # continuation in phi_inf through the capillary-bridging limit points, and
    # the 1D backward Fokker-Planck first-passage-time solver for the
    # enhancer-promoter distance.
    include("bvp_two_spheres.jl")
    include("bvp_arclength.jl")
    include("fpt_solver.jl")

    export Parameters, with, fb, dfb_dphi, d2fb_dphi2, psi_linear,
           compute_mu_inf, compute_spinodal, is_spherical,
           IsolatedSphere1D, solve_isolated_sphere,
           PlanarSlab1D, solve_parallel_plates, psi_linear_plates,
           excess_free_energy, plates_pair_potential,
           Geometry, build_geometry, decay_length,
           scatter, scatter!, laplacian, laplacian!,
           make_residual, make_residual_inplace, build_linear_operator,
           initial_guess, solve_two_spheres,
           interp_phi, cell_fluid_fraction,
           phi_on_sphere, excess_free_energy_two_spheres,
           quadratic_ghost_coefficients, find_clean_stencil,
           ArcTrace, arclength_phi_inf, polish_at_phi_inf, gap_center_phi,
           FPTProblem, solve_fpt, boltzmann_ic, fpt_mfpt_quadrature

end # module Droplet
