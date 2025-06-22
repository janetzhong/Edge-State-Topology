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

## Citation

If you use this code or build on this work, please consider citing us:

```bibtex
@misc{zhong2025topological,
      title={Topological nature of edge states for one-dimensional systems without symmetry protection},
      author={Janet Zhong and Heming Wang and Alexander N Poddubny and Shanhui Fan},
      year={2025},
      eprint={2412.10526},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mes-hall},
      url={https://arxiv.org/abs/2412.10526},
}
```
