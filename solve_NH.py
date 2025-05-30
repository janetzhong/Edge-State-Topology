"""
solve_NH.py
"""

import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from utils import *

plt.rcParams.update({
    'font.size': 23,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def main():
    start_time = time.time()

    steps = 80
    t_values = np.linspace(-1.5, 1.5, steps)
    var_values = np.linspace(0, 0.9, steps)

    base_dir, csv_file_path, output_figures_dir = create_output_directory("NH")
    output_results = []
    figure_id = 1
    will_save = True

    for i, t in enumerate(t_values):
        for j, var in enumerate(var_values):

            figure_path = f"{output_figures_dir}/figure_{figure_id}.jpg"
            if will_save and os.path.exists(figure_path):
                print(f"skipping {figure_id}")
                figure_id += 1
                continue

            # Construct NH Hamiltonian
            hminus, hzero, hplus = create_NH_matrix(t, var)
            H = create_Hobc(hminus, hzero, hplus)
            is_hermitian = isHermitian(H)

            # Solve
            eigenvalues, eigenvectors = np.linalg.eig(H)
            Edeg1, Edeg2 = solve_Edeg(hminus, hzero, hplus)
            Edeg_list = [Edeg1, Edeg2]
            Eedge, Enotedge, Nedge = solve_Eedge_Enotedge_Nedge(Edeg_list, hminus, hzero, hplus)
            edge_indicators, left_edge_indicator, right_edge_indicator = solve_edge_indicators(eigenvalues, hminus, hzero, hplus)
            zgbzplus, zgbzminus, mgbzplus, mgbzminus = solve_gbz_NH(eigenvalues, Eedge, hminus, hzero, hplus)
            zbranchpts = solve_zbranch_pts(hminus, hzero, hplus)
            Mbranchlist = solve_Mbranch_pts(hminus, hzero, hplus)
            Mdeg1, Mdeg2 = solve_Mdeg(Edeg_list, hminus, hzero, hplus, tol=1e-3)

            # Calculate winding numbers
            winding_data = calculate_winding_number_NH(
                zgbzplus, zgbzminus,
                mgbzplus, mgbzminus,
                [Mdeg1, Mdeg2],
                Mbranchlist
            )
            Wdeg1 = winding_data['Wdeg1']
            Wdeg2 = winding_data['Wdeg2']
            Wtotal = winding_data['Wtotal']
            Mdeg1_count = winding_data['Mdeg1_count']
            Mdeg2_count = winding_data['Mdeg2_count']
            Mb_count = winding_data['Mb_count']
            interp_plus, interp_minus = winding_data['interp_curves']
            interp_real_plus = interp_plus[0]
            interp_imag_plus = interp_plus[1]
            interp_real_minus = interp_minus[0]
            interp_imag_minus = interp_minus[1]

            # Create figure
            create_main_figure_NH(
                eigenvalues, eigenvectors, Eedge, Enotedge,
                zgbzplus, zgbzminus, zbranchpts,
                mgbzplus, mgbzminus, edge_indicators,
                left_edge_indicator, right_edge_indicator,
                np.array(Mbranchlist), Mdeg1, Mdeg2,
                interp_real_plus, interp_imag_plus,
                interp_real_minus, interp_imag_minus,
                t, var, is_hermitian,
                Wdeg1, Wdeg2,
                Mdeg1_count, Mdeg2_count, Mb_count
            )

            output_variables = [t, var, len(Eedge), Wtotal, figure_id, left_edge_indicator, right_edge_indicator]
            output_results.append(output_variables)

            if will_save:
                plt.savefig(figure_path)
                plt.close()
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(output_variables)
                    
            print(figure_id)
            figure_id += 1

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.5f} seconds")

if __name__ == '__main__':
    main()
