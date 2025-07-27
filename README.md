# Topological nature of edge states for one-dimensional systems without symmetry protection

This repository contains the code to reproduce the topological invariant phase diagrams in our paper.

## How to Use

To solve for Hermitian phase diagram, run:

```bash
python solve_H.py
```

To solve for non-Hermitian phase diagram, run:

```bash
python solve_NH.py
```

Each run creates a timestamped folder in `results/` containing:

- `output_variables.csv` – computed data. Pre-computed data has also been uploaded in this repository.
- `output_figures/` – auto-generated plots per model (git-ignored except for one example figure).

To visualize full phase diagrams, set the `foldername` variable in `load_phase_diagram.py`, then run:

```bash
python load_phase_diagram.py
```

This creates a `phase_diag_figures/` folder inside the same results directory.

## Extra

- `edgestate_topology_symbolic.nb`: We also include a starter Mathematica notebook for symbolic formulas and figures.

## Citation

If you use this code or build on this work, please consider citing us:

```bibtex
@article{zhong2025topological,
  title = {Topological Nature of Edge States for One-Dimensional Systems without Symmetry Protection},
  author = {Zhong, Janet and Wang, Heming and Poddubny, Alexander N. and Fan, Shanhui},
  journal = {Phys. Rev. Lett.},
  volume = {135},
  issue = {1},
  pages = {016601},
  numpages = {10},
  year = {2025},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/k77w-ft26},
  url = {https://link.aps.org/doi/10.1103/k77w-ft26}
}
```
