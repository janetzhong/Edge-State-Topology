"""
solve_H.py
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

    base_dir, csv_file_path, output_figures_dir = create_output_directory("H")
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

            # Construct Hamiltonian
            hminus, hzero, hplus = create_H_matrix(t, var)
            H = create_Hobc(hminus, hzero, hplus)
            is_hermitian = isHermitian(H)

            # Solve
            eigenvalues, eigenvectors = np.linalg.eig(H)
            all_edge_indicators, left_edge_indicator, right_edge_indicator = solve_edge_indicators(eigenvalues, hminus, hzero, hplus)
            Edeg_list = solve_Edeg(hminus, hzero, hplus)
            Eedge, Enotedge, Nedge = solve_Eedge_Enotedge_Nedge(Edeg_list, hminus, hzero, hplus)
            zbz, mbzplus, mbzminus = solve_bz_H(hminus, hzero, hplus)
            Mdeg1, Mdeg2 = solve_Mdeg_analytical(hminus, hzero, hplus)
            zbranchpts = solve_zbranch_pts(hminus, hzero, hplus)

            # Calculate winding numbers
            winding_data = calculate_winding_numbers_H([Mdeg1, Mdeg2], zbz, mbzplus, mbzminus)
            Wtotal = winding_data['Wtotal']
            Wbz1 = winding_data['Wbz1']
            Wbz2 = winding_data['Wbz2']
            interp_real_plus_closed, interp_imag_plus_closed = winding_data['interp_curves'][0]
            interp_real_minus_closed, interp_imag_minus_closed = winding_data['interp_curves'][1]

            # Create figure
            create_main_figure_H(
                eigenvalues, eigenvectors, Eedge, Enotedge,
                zbz, zbranchpts,
                mbzplus, mbzminus, all_edge_indicators,
                left_edge_indicator, right_edge_indicator,
                Mdeg1, Mdeg2,
                interp_real_plus_closed, interp_imag_plus_closed,
                interp_real_minus_closed, interp_imag_minus_closed,
                t, var, is_hermitian, Wbz1, Wbz2
            )

            output_variables = [t, var, Nedge, Wtotal, figure_id, left_edge_indicator, right_edge_indicator]
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
