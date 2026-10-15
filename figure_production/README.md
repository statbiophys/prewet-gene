# Paper figures

Each `figN/` folder holds the notebook that draws the figure, the scripts that
produce its data, and the resulting PDF. `processed_data/` holds the reduced
data of all figures.

## Requirements

- Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`, `pypdf`, and `cairosvg`
  (or Inkscape).
- LaTeX with `cm-super` (the figures use `text.usetex`).
- To regenerate the data: Julia 1.11.1 (see `../README.md`), then, once:

  ```
  julia --project=@. -e 'using Pkg; Pkg.instantiate()'
  ```

## Drawing the figures

The committed `processed_data/` is enough. Run the notebook of each figure:

```
jupyter lab figure_production/fig1/fig1.ipynb      # -> fig1/fig1.pdf
```

and likewise `fig2` to `fig6` and `figs1`.

Fig. 4's cartoon is converted with `cairosvg`.

## Regenerating the data

From the repository root. Simulations write to `data/`; the `prepare_*` scripts
write `processed_data/`.

**Figs. 1–3** (cluster: 16 concentrations × 16 runs of ~3 days). Use the same
`DATE` for every part of the sweep:

```
DATE=YYYYMMDD bash figure_production/run_lattice_sweep.sh all
python figure_production/fig1/prepare_fig1_data.py --date YYYYMMDD
python figure_production/fig2/prepare_fig2_data.py --date YYYYMMDD
python figure_production/fig3/prepare_fig3_data.py --date YYYYMMDD
```

**Fig. 4 and Fig. S1** (minutes):

```
bash figure_production/fig4/run_fig4_solver.sh
python figure_production/fig4/prepare_fig4_data.py
```

**Fig. 5** (minutes):

```
bash figure_production/fig5/run_fig5_rouse.sh
python figure_production/fig5/prepare_fig5_data.py
```

**Fig. 6** (cluster: hours to a day per process). Run the stages in this order,
with the same `DATE`:

```
export DATE=YYYYMMDD
bash figure_production/fig6/run_fig6.sh bvp
bash figure_production/fig6/run_fig6.sh sweep
bash figure_production/fig6/run_fig6.sh potential
bash figure_production/fig6/run_fig6.sh s300
bash figure_production/fig6/run_fig6.sh s_scan
python figure_production/fig6/prepare_fig6_data.py
```

`RUN_MODE=smoke` runs a short test of the `s300` and `s_scan` stages. The
script's header lists the options (number of processes, output folders).
