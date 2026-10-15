# prewet-gene

Simulation code for a model of transcription regulated by ligand condensation
on DNA. Four codes, all in the Julia package `Droplet` (`src/`):

1. **Lattice Monte Carlo** — a DNA polymer with enhancer sites and diffusing ligands on a cubic lattice.
2. **Surface condensation solver** — equilibrium ligand profile around an attractive sphere (Flory–Huggins + gradient term).
3. **Rouse polymer** — overdamped bead–spring chain with a pair force between two beads.
4. **Two spheres and first passage** — ligand profile between two attractive spheres, and first-passage times in the resulting potential.

**To reproduce the paper's figures, see [`figure_production/README.md`](figure_production/README.md).**
The reduced data are committed there, so the figures can be redrawn without running any simulation;
the commands to regenerate both the raw and the reduced data are also given there.

## Setup

```bash
juliaup add 1.11.1 && juliaup default 1.11.1          # Manifest.toml is pinned to Julia 1.11.1
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pip install -r requirements.txt                       # Python, for data reduction and figures
```

Run everything from the repository root; simulation output goes to `data/` (not committed).

## 1. Lattice Monte Carlo

Code: `types.jl`, `interactions.jl`, `hamiltonian.jl`, `metropolis.jl`, `move.jl`, `init.jl`, `stats.jl`, `port.jl`.
A run is defined by a spec file (`spec/lattice/proximal_promoter_transcription.jl`) that sets the
parameter list `alphas` and a function `target!` returning one run's measurements.

```bash
# generic: run spec SPEC split over N processes, output OUT/output_cpu<k>.json
./runJL sim/sweep.jl SPEC N OUT

# one ligand volume fraction of the sweep, 16 runs of T_MAX minutes
T_MAX=5 DATE=20261013 bash figure_production/run_lattice_sweep.sh 0.006 4   # short test, 4 at a time
```

Runs are unseeded, so a rerun reproduces statistics, not trajectories.

## 2. Surface condensation solver

Code: `bvp_solver.jl`.

```julia
using Droplet
p   = Parameters(nb = 2230, chi_b = 0.5448, kappa = 10.12, omega_b = -0.034,
                 R = 66.66, L = 200, phi_inf = 5e-5)      # parameters of spec/fig4/20261004/small_R
sol = solve_isolated_sphere(p; L_iso = p.L, N = 400, track_branches = true)
sol.z, sol.phi, sol.branch            # distance from the sphere surface, profile, :thin/:thick/:unique
psi, dpsi, lam = psi_linear(sol.z, p) # linear-response profile and its decay rate
p2  = with(p; phi_inf = 7e-5)         # copy with one parameter changed
```

Sweeps over φ∞ are driven by `spec/fig4/<date>/<run>/demo_bvp.jl`, each with a `param.jl` holding all parameters:
`julia --project=. spec/fig4/20261004/small_R/demo_bvp.jl`.

## 3. Rouse polymer

Code: `rouse.jl`.

```julia
using Droplet
r = RousePolymer(; nbeads = 100, kuhnLength = 1.0, friction = 1.0, dt = 1e-3, nsteps = 10^6,
                   trackedBead1 = 1, trackedBead2 = 100, r_min = 2.5,
                   attractiveForce = d -> -0.5 * (d - 5.0),   # f(d) = -dU/dd
                   saveMode = :distance_only, seed = 1)
init!(r); simulate!(r)
r.distanceTraj                        # distance between the two tracked beads
```

Example spec: `figure_production/fig5/rouse_toy.jl`, run with `bash figure_production/fig5/run_fig5_rouse.sh`.
Also used for Fig. 6: `spec/fig6/*/rouse_mfpt_attractive.jl`.

## 4. Two spheres and first passage

Code: `bvp_two_spheres.jl` (axisymmetric 2D solve), `bvp_arclength.jl` (continuation in φ∞), `fpt_solver.jl`.

```julia
using Droplet
res = solve_two_spheres(p, 20.0; Nrho = 80, Nz = 200)   # p as in section 2; 20.0 = gap between sphere surfaces
res[:phi], res[:success]
excess_free_energy_two_spheres(res)

prob = FPTProblem(x, U; D = 1.0, left_bc = :reflecting, right_bc = :absorbing)   # U sampled on uniform grid x
sol  = solve_fpt(prob)
sol.mfpt, sol.t, sol.pdf              # mean first-passage time, first-passage-time density
```

Pipeline: `bash figure_production/fig6/run_fig6.sh bvp|sweep|potential|s300|s_scan|all`, specs in `spec/fig6/`.
