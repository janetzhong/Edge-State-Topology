"""
load_phase_diagram.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
plt.rcParams.update({
    'font.size': 40,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# set output folder
foldername = "NH_2025-06-02_09-36"
foldername = "H_2025-06-02_09-36"

# set variables and load data
steps = 80
t_vals = np.linspace(-1.5, 1.5, steps)
a_vals = np.linspace(0, 0.9, steps)
cwd = os.getcwd()
path = os.path.join(cwd, "results", foldername)
figdir = os.path.join(path, "phase_diag_figures")
os.makedirs(figdir, exist_ok=True)
df = pd.read_csv(os.path.join(path, "output_variables.csv"))
output_results = df.to_numpy().reshape(len(t_vals), len(a_vals), -1)

# analytical edge states
plt.figure()
plt.imshow(output_results[:, :, 2].T, extent=[t_vals.min(), t_vals.max(), a_vals.max(), a_vals.min()],
           aspect='auto', norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), cmap=plt.cm.get_cmap('viridis', 3))
plt.colorbar(ticks=[0, 1, 2]).set_label(r'$N_{edge}$')
plt.xlabel(r'$t$'); plt.ylabel(r'$\alpha$')
plt.title(r'$\mathrm{analytical\ edge\ states}$', pad=20)
plt.yticks([a_vals.min(), a_vals.mean(), a_vals.max()])
plt.savefig(os.path.join(figdir, "analytical_edge_states.png"), dpi=300, bbox_inches='tight')

# left edge indicator
plt.figure()
plt.imshow(np.log(np.abs(output_results[:, :, 5].T) + 1e-15), extent=[t_vals.min(), t_vals.max(), a_vals.max(), a_vals.min()],
           aspect='auto', vmin=-3, vmax=1)
plt.colorbar().set_label(r'$\log(|\mathcal{I}_{\mathrm{edge}}|)$')
plt.xlabel(r'$t$'); plt.ylabel(r'$\alpha$')
plt.title(r"$\mathrm{left\ edge\ indicator}$", pad=30)
plt.yticks([a_vals.min(), a_vals.mean(), a_vals.max()])
plt.savefig(os.path.join(figdir, "left_edge_indicator.png"), dpi=300, bbox_inches='tight')

# right edge indicator
plt.figure()
plt.imshow(np.log(np.abs(output_results[:, :, 6].T) + 1e-15), extent=[t_vals.min(), t_vals.max(), a_vals.max(), a_vals.min()],
           aspect='auto', vmin=-3, vmax=1)
plt.colorbar().set_label(r'$\log(|\mathcal{I}_{\mathrm{edge}}|)$')
plt.xlabel(r'$t$'); plt.ylabel(r'$\alpha$')
plt.title(r"$\mathrm{right\ edge\ indicator}$", pad=30)
plt.yticks([a_vals.min(), a_vals.mean(), a_vals.max()])
plt.savefig(os.path.join(figdir, "right_edge_indicator.png"), dpi=300, bbox_inches='tight')

# topological invariant
plt.figure()
plt.imshow(output_results[:, :, 3].T, extent=[t_vals.min(), t_vals.max(), a_vals.max(), a_vals.min()],
           aspect='auto', norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), cmap=plt.cm.get_cmap('viridis', 3))
plt.colorbar(ticks=[0, 1, 2]).set_label(r'$W$')
plt.xlabel(r'$t$'); plt.ylabel(r'$\alpha$')
plt.title(r"$\mathrm{topological\ invariant}$", pad=20)
plt.yticks([a_vals.min(), a_vals.mean(), a_vals.max()])
plt.savefig(os.path.join(figdir, "topological_invariant.png"), dpi=300, bbox_inches='tight')
