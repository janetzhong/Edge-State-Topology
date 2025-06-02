from datetime import datetime
from pathlib import Path as FilePath
import csv
import numpy as np
from numpy import isclose
from scipy.interpolate import splprep, splev
from matplotlib.path import Path
import cmath
from collections import Counter
import matplotlib.pyplot as plt


def create_output_directory(prefix=""):
    now = datetime.now()
    base_name = f"{prefix + '_' if prefix else ''}{now:%Y-%m-%d_%H-%M}"
    base_dir = FilePath(f"./results/{base_name}")
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = base_dir / "output_variables.csv"
    with csv_file_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            't', 'var', 'len_analytical_edge_states',
            'Wtotal', 'figure_id', 'left_edge_indicator', 'right_edge_indicator'
        ])
    figures_dir = base_dir / "output_figures"
    figures_dir.mkdir(exist_ok=True)
    return base_dir, csv_file_path, figures_dir


def create_NH_matrix(t, var):
    tprime = 1
    # SSH matrix
    hsshMinus = np.array([[0, tprime], [0, 0]])
    hsshZero = np.array([[0, t], [t, 0]])
    hsshPlus = np.array([[0, 0], [tprime, 0]])
    # Non-Hermitian random matrix Hnh
    hnrandminus = np.array([[0.74 - 0.13j, -0.79 - 0.84j], [-0.43 - 0.08j, 0.01 + 0.6j]])
    hnrandzero = np.array([[-0.18 + 0.32j, -0.44 - 0.84j], [-0.33 + 0.7j, 0.21 - 0.74j]])
    hnrandplus = np.array([[0.51 + 0.4j, -0.86 + 0.4j], [-0.44 + 0.38j, -0.86 - 0.03j]])
    # Combined matrices
    hminus = (1 - var) * hnrandminus + var * hsshMinus
    hzero = (1 - var) * hnrandzero + var * hsshZero
    hplus = (1 - var) * hnrandplus + var * hsshPlus
    return hminus, hzero, hplus

def create_H_matrix(t, var):
    tprime = 1
    # SSH matrix
    hsshMinus = np.array([[0, tprime], [0, 0]])
    hsshZero = np.array([[0, t], [t, 0]])
    hsshPlus = np.array([[0, 0], [tprime, 0]])
    # Hermitian random matrix Hh
    hrandMinus = np.array([[0.84 - 0.14j, 0.02 - 0.69j], [0.02 - 0.69j, -0.82 - 0.60j]])
    hrandZero = np.array([[-0.66 + 0.00j, -0.96 - 0.80j], [-0.96 + 0.80j, -0.58 + 0.00j]])
    hrandPlus = np.array([[0.84 + 0.14j, 0.02 + 0.69j], [0.02 + 0.69j, -0.82 + 0.60j]])
    # Combined matrices
    hminus = (1 - var) * hrandMinus + var * hsshMinus
    hzero = (1 - var) * hrandZero + var * hsshZero
    hplus = (1 - var) * hrandPlus + var * hsshPlus
    return hminus, hzero, hplus

def create_Hobc(hminus, hzero, hplus, nunitcell=32, nsublattice=2):
    H = np.zeros((nunitcell * nsublattice, nunitcell * nsublattice), dtype=complex)
    for i in range(nunitcell):
        for j in range(nunitcell):
            shift = j - i
            if shift == 0:
                H[i*nsublattice:(i+1)*nsublattice, j*nsublattice:(j+1)*nsublattice] = hzero
            elif shift == 1:
                H[i*nsublattice:(i+1)*nsublattice, j*nsublattice:(j+1)*nsublattice] = hplus
            elif shift == -1:
                H[i*nsublattice:(i+1)*nsublattice, j*nsublattice:(j+1)*nsublattice] = hminus
    return H

def isHermitian(H, tol=1e-10):
    return np.allclose(H, np.conj(H.T), atol=tol)

def calc_M_eqn1(z, E, hminus, hzero, hplus):
    tabM1 = hminus[0, 1]
    tab0  = hzero[0, 1]
    tabP1 = hplus[0, 1]
    taaM1 = hminus[0, 0]
    taa0  = hzero[0, 0]
    taaP1 = hplus[0, 0]
    numerator   = tab0 + tabM1 / z + tabP1 * z
    denominator = -taa0 + E - taaM1 / z - taaP1 * z
    return numerator / denominator

def find_z1toz4_M1toM4(E, hminus, hzero, hplus):
    taaM1 = hminus[0, 0]
    tabM1 = hminus[0, 1]
    tbaM1 = hminus[1, 0]
    tbbM1 = hminus[1, 1]
    taa0  = hzero[0, 0]
    tab0  = hzero[0, 1]
    tba0  = hzero[1, 0]
    tbb0  = hzero[1, 1]
    taaP1 = hplus[0, 0]
    tabP1 = hplus[0, 1]
    tbaP1 = hplus[1, 0]
    tbbP1 = hplus[1, 1]
    # Polynomial coefficients [z^4, z^3, z^2, z^1, z^0]
    c4 = -tabP1 * tbaP1 + taaP1 * tbbP1
    c3 = -tabP1 * tba0 - tab0 * tbaP1 + taaP1 * tbb0 + taa0 * tbbP1 - taaP1 * E - tbbP1 * E
    c2 = (-tab0 * tba0 - tabP1 * tbaM1 - tabM1 * tbaP1 +
          taa0 * tbb0 + taaP1 * tbbM1 + taaM1 * tbbP1 -
          taa0 * E - tbb0 * E + E ** 2)
    c1 = -tabM1 * tba0 - tab0 * tbaM1 + taaM1 * tbb0 + taa0 * tbbM1 - taaM1 * E - tbbM1 * E
    c0 = -tabM1 * tbaM1 + taaM1 * tbbM1
    roots = np.roots([c4, c3, c2, c1, c0])
    roots_sorted = sorted(roots, key=abs)
    z1, z2, z3, z4 = roots_sorted
    M1 = calc_M_eqn1(z1, E, hminus, hzero, hplus)
    M2 = calc_M_eqn1(z2, E, hminus, hzero, hplus)
    M3 = calc_M_eqn1(z3, E, hminus, hzero, hplus)
    M4 = calc_M_eqn1(z4, E, hminus, hzero, hplus)
    return (z1, z2, z3, z4), (M1, M2, M3, M4)

def solve_edge_indicators(eigenvalues, hminus, hzero, hplus):
    all_edge_indicators = []
    for E in eigenvalues:
        try:
            (_, z2, z3, _), (M1, M2, M3, M4) = find_z1toz4_M1toM4(E, hminus, hzero, hplus)
            diff_left = abs(M1 - M2)
            diff_right = abs(M3 - M4)
            log_term = np.log(abs(z3) / abs(z2))
            indicator = -log_term if diff_left < diff_right else log_term
            all_edge_indicators.append(indicator)
        except Exception:
            continue
    all_edge_indicators = np.array(all_edge_indicators)
    left_edge_indicator = all_edge_indicators.min()
    right_edge_indicator = all_edge_indicators.max()
    return all_edge_indicators, left_edge_indicator, right_edge_indicator

def solve_Edeg(hminus, hzero, hplus):
    tr = lambda H: np.trace(H) / 2
    tl = lambda H: H - tr(H) * np.eye(2)
    ip = lambda A, B: 0.5 * np.trace(A @ B)
    dp, d0, dm = tr(hplus), tr(hzero), tr(hminus)
    hp, h0, hm = tl(hplus), tl(hzero), tl(hminus)
    G = [[ip(hp, hp), ip(hp, hm)], [ip(hm, hp), ip(hm, hm)]]
    D = [[dp, d0, dm], [ip(hp, hp), ip(hp, h0), ip(hp, hm)], [ip(hm, hp), ip(hm, h0), ip(hm, hm)]]
    T = [[ip(hp, hp), ip(hp, h0), ip(hp, hm)], [ip(h0, hp), ip(h0, h0), ip(h0, hm)], [ip(hm, hp), ip(hm, h0), ip(hm, hm)]]
    v = tl(dp * hminus - dm * hplus)
    corr = ip(v, v)
    detG, detD, detT = map(np.linalg.det, [G, D, T])
    s = np.sqrt(detT * (detG - corr))
    Edeg1 = (-detD + s) / detG
    Edeg2 = (-detD - s) / detG
    return (Edeg1, Edeg2)

def solve_Eedge_Enotedge_Nedge(Edeg_list, hminus, hzero, hplus, tol=1e-3):
    Eedge, Enotedge = [], []
    for E in Edeg_list:
        (_, _, _, _), (M1, M2, M3, M4) = find_z1toz4_M1toM4(E, hminus, hzero, hplus)
        r12 = abs(M1 / M2 - 1)
        r34 = abs(M3 / M4 - 1)
        if r12 <= tol or r34 <= tol:
            Eedge.append(E)
        else:
            Enotedge.append(E)
    Nedge = len(Eedge)
    return Eedge, Enotedge, Nedge

def solve_Mdeg_analytical(hminus, _, hplus):
    a = hplus[0,0]*hminus[1,0] - hminus[0,0]*hplus[1,0]
    b = hplus[0,1]*hminus[1,0] - hminus[0,1]*hplus[1,0] + hplus[0,0]*hminus[1,1] - hminus[0,0]*hplus[1,1]
    c = hplus[0,1]*hminus[1,1] - hminus[0,1]*hplus[1,1]
    roots = np.roots([a, b, c])
    Mdega, Mdegb = roots
    return Mdega, Mdegb

def solve_Mdeg(Edeg_list, hminus, hzero, hplus, tol=1e-3):
    deg_M_list = []
    for E in Edeg_list:
        _, (M1, M2, M3, M4) = find_z1toz4_M1toM4(E, hminus, hzero, hplus)
        if abs(M1/M2 - 1) <= tol:
            deg_M_list.append(M1)
        elif abs(M1/M3 - 1) <= tol:
            deg_M_list.append(M1)
        elif abs(M1/M4 - 1) <= tol:
            deg_M_list.append(M1)
        elif abs(M2/M3 - 1) <= tol:
            deg_M_list.append(M2)
        elif abs(M2/M4 - 1) <= tol:
            deg_M_list.append(M2)
        elif abs(M3/M4 - 1) <= tol:
            deg_M_list.append(M3)
        else:
            # fallback: should not need this though
            deg_M_list.append(0 + 0j)
    # Ensure two outputs
    if len(deg_M_list) < 2:
        deg_M_list += [deg_M_list[-1]] * (2 - len(deg_M_list))
    return tuple(deg_M_list[:2])

def Eplus(z, hminus, hzero, hplus):
    get = lambda M, i, j: M[i, j]
    terms = lambda i, j: get(hminus, i, j)/z + get(hzero, i, j) + get(hplus, i, j)*z
    do = 0.5 * (terms(0, 0) + terms(1, 1))
    dx = 0.5 * (terms(0, 1) + terms(1, 0))
    dy = 0.5j * (terms(0, 1) - terms(1, 0))
    dz = 0.5 * (terms(0, 0) - terms(1, 1))
    return do + np.sqrt(dx**2 + dy**2 + dz**2)

def Eminus(z, hminus, hzero, hplus):
    get = lambda M, i, j: M[i, j]
    terms = lambda i, j: get(hminus, i, j)/z + get(hzero, i, j) + get(hplus, i, j)*z
    do = 0.5 * (terms(0, 0) + terms(1, 1))
    dx = 0.5 * (terms(0, 1) + terms(1, 0))
    dy = 0.5j * (terms(0, 1) - terms(1, 0))
    dz = 0.5 * (terms(0, 0) - terms(1, 1))
    return do - np.sqrt(dx**2 + dy**2 + dz**2)

def solve_gbz_NH(eigenvalues, Eedge, hminus, hzero, hplus, tol=1e-3):
    # Note: this gbz solver (keeping middle two roots) generally works for degree 4 in z characteristic 
    # polynomial. However, this subGBZ sorter using Eplus and Eminus functions works for the case in our 
    # paper, but does NOT work for general NH models. For more general cases one should use other 
    # analytical or machine learning methods to sort subGBZ.
    def filter_eigenvalues_closest(eigenvalues, Eedge):
        eigenvalues = np.array(eigenvalues)
        indices_to_remove = []
        for aes in Eedge:
            differences = np.abs(eigenvalues - aes)
            min_diff_index = np.argmin(differences)
            if min_diff_index not in indices_to_remove:
                indices_to_remove.append(min_diff_index)
        return np.delete(eigenvalues, indices_to_remove)
    gbzplus, gbzminus = [], []
    mgbzplus, mgbzminus = [], []
    # Remove the closest eigenvalue to each Eedge
    filtered_eigenvalues = filter_eigenvalues_closest(eigenvalues, Eedge)
    for eig in filtered_eigenvalues:
        try:
            (_, z2, z3, _), _ = find_z1toz4_M1toM4(eig, hminus, hzero, hplus)
            for z in [z2, z3]: 
                M = calc_M_eqn1(z, eig, hminus, hzero, hplus)
                if isclose(Eplus(z, hminus, hzero, hplus), eig, rtol=1e-6, atol=1e-10):
                    gbzplus.append(z)
                    mgbzplus.append(M)
                elif isclose(Eminus(z, hminus, hzero, hplus), eig, rtol=1e-6, atol=1e-10):
                    gbzminus.append(z)
                    mgbzminus.append(M)
        except Exception:
            continue
    return gbzplus, gbzminus, mgbzplus, mgbzminus


def solve_bz_H(hminus, hzero, hplus, bzsteps=500):
    # for Hermitian models, the GBZ is just the Brillouin zone unit circle on the z plane
    angles = np.linspace(0, 2 * np.pi, bzsteps)
    zbz = [np.exp(1j * theta) for theta in angles]
    mbzplus = []
    mbzminus = []
    for z in zbz:
        Eplus_val = Eplus(z, hminus, hzero, hplus)
        Eminus_val = Eminus(z, hminus, hzero, hplus)
        # This method of sorting the two M(subGBZ) loops works for the case in our paper, but 
        # again is not general. 
        mbzplus.append(calc_M_eqn1(z, Eplus_val, hminus, hzero, hplus))
        mbzminus.append(calc_M_eqn1(z, Eminus_val, hminus, hzero, hplus))
    return zbz, mbzplus, mbzminus

def solve_zbranch_pts(hminus, hzero, hplus):
    taaM1, tabM1, tbaM1, tbbM1 = hminus[0, 0], hminus[0, 1], hminus[1, 0], hminus[1, 1]
    taa0,  tab0,  tba0,  tbb0  = hzero[0, 0],  hzero[0, 1],  hzero[1, 0],  hzero[1, 1]
    taaP1, tabP1, tbaP1, tbbP1 = hplus[0, 0], hplus[0, 1], hplus[1, 0], hplus[1, 1]
    # Coefficients of z^0 to z^4
    c0 = (taaM1**2)/4 + tabM1*tbaM1 - (taaM1*tbbM1)/2 + (tbbM1**2)/4
    c1 = (taa0*taaM1)/2 + tabM1*tba0 + tab0*tbaM1 - (taaM1*tbb0)/2 - (taa0*tbbM1)/2 + (tbb0*tbbM1)/2
    c2 = (taa0**2)/4 + 0.5*taaM1*taaP1 + tab0*tba0 + tabP1*tbaM1 + tabM1*tbaP1 \
         - 0.5*taa0*tbb0 + (tbb0**2)/4 - 0.5*taaP1*tbbM1 - 0.5*taaM1*tbbP1 + 0.5*tbbM1*tbbP1
    c3 = 0.5*taa0*taaP1 + tabP1*tba0 + tab0*tbaP1 - 0.5*taaP1*tbb0 - 0.5*taa0*tbbP1 + 0.5*tbb0*tbbP1
    c4 = (taaP1**2)/4 + tabP1*tbaP1 - 0.5*taaP1*tbbP1 + (tbbP1**2)/4
    zbranch = np.roots([c4, c3, c2, c1, c0])
    return zbranch

def solve_Mbranch_pts(hminus, hzero, hplus):
    taaM1, tabM1, tbaM1, tbbM1 = hminus[0, 0], hminus[0, 1], hminus[1, 0], hminus[1, 1]
    taa0,  tab0,  tba0,  tbb0  = hzero[0, 0],  hzero[0, 1],  hzero[1, 0],  hzero[1, 1]
    taaP1, tabP1, tbaP1, tbbP1 = hplus[0, 0], hplus[0, 1], hplus[1, 0], hplus[1, 1]
    # Coefficients for M^0 to M^4
    coeffs = [
        tba0**2 - 4 * tbaM1 * tbaP1,  # M^4
        -2 * taa0 * tba0 + 4 * taaP1 * tbaM1 + 4 * taaM1 * tbaP1 +
        2 * tba0 * tbb0 - 4 * tbaP1 * tbbM1 - 4 * tbaM1 * tbbP1,  # M^3
        taa0**2 - 4 * taaM1 * taaP1 - 2 * tab0 * tba0 + 4 * tabP1 * tbaM1 +
        4 * tabM1 * tbaP1 - 2 * taa0 * tbb0 + tbb0**2 +
        4 * taaP1 * tbbM1 + 4 * taaM1 * tbbP1 - 4 * tbbM1 * tbbP1,  # M^2
        2 * taa0 * tab0 - 4 * taaP1 * tabM1 - 4 * taaM1 * tabP1 -
        2 * tab0 * tbb0 + 4 * tabP1 * tbbM1 + 4 * tabM1 * tbbP1,  # M^1
        tab0**2 - 4 * tabM1 * tabP1  # M^0
    ]
    Mbranch = np.roots(coeffs)
    return Mbranch


def interpolate_curve(mvals):
    # Remove near-duplicates
    filtered = []
    seen = set()
    for m in mvals:
        key = (round(m.real, 8), round(m.imag, 8))
        if key not in seen:
            filtered.append(m)
            seen.add(key)
    # Require at least 4 points for cubic spline
    if len(filtered) < 4:
        raise ValueError("Too few unique points for spline interpolation")
    real = [m.real for m in filtered]
    imag = [m.imag for m in filtered]
    # Force loop closure
    if abs(filtered[0] - filtered[-1]) > 1e-6:
        real.append(real[0])
        imag.append(imag[0])
    tck, _ = splprep([real, imag], s=0, per=True)
    u = np.linspace(0, 1, 200)
    real_interp, imag_interp = splev(u, tck)
    return np.append(real_interp, real_interp[0]), np.append(imag_interp, imag_interp[0])


def calculate_winding_numbers_H(Mdeglist, zbz, mbzplus, mbzminus):
    Mdeg1, Mdeg2 = Mdeglist
    # Sort by arg(z)
    sorted_indices = sorted(range(len(zbz)), key=lambda i: cmath.phase(zbz[i]))
    mbzplus_sorted = [mbzplus[i] for i in sorted_indices]
    mbzminus_sorted = [mbzminus[i] for i in sorted_indices]
    # Interpolated closed curves for both subGBZs
    interp_plus = interpolate_curve(mbzplus_sorted)
    interp_minus = interpolate_curve(mbzminus_sorted)
    path_plus = Path(np.column_stack(interp_plus))
    path_minus = Path(np.column_stack(interp_minus))
    def count_inside(path, point):
        return int(path.contains_point((point.real, point.imag)))
    # Compute winding numbers
    Wbz1 = count_inside(path_plus, Mdeg1) + count_inside(path_plus, Mdeg2)
    Wbz2 = count_inside(path_minus, Mdeg1) + count_inside(path_minus, Mdeg2)
    return {
        'Wtotal': Wbz1 + Wbz2,
        'Wbz1': Wbz1,
        'Wbz2': Wbz2,
        'interp_curves': (interp_plus, interp_minus)
    }


def calculate_winding_number_NH(gbzplus, gbzminus, mgbzplus, mgbzminus, Mdeglist, Mbranchlist):
    Mdeg1, Mdeg2 = Mdeglist
    # Sort by phase
    sort_by_phase = lambda zlist, mlist: [mlist[zlist.index(z)] for z in sorted(zlist, key=cmath.phase)]
    mgbzplus_sorted = sort_by_phase(gbzplus, mgbzplus)
    mgbzminus_sorted = sort_by_phase(gbzminus, mgbzminus)
    # Interpolated closed curves for both subGBZs
    interp_plus = interpolate_curve(mgbzplus_sorted)
    interp_minus = interpolate_curve(mgbzminus_sorted)
    path_plus = Path(np.column_stack(interp_plus))
    path_minus = Path(np.column_stack(interp_minus))
    # Count
    def contour_count(point):
        inside_plus = path_plus.contains_point(point)
        inside_minus = path_minus.contains_point(point)
        return 2 if inside_plus and inside_minus else 1 if inside_plus or inside_minus else 0
    Mdeg1_count = contour_count((Mdeg1.real, Mdeg1.imag))
    Mdeg2_count = contour_count((Mdeg2.real, Mdeg2.imag))
    Mb_counts = [contour_count((Mb.real, Mb.imag)) for Mb in Mbranchlist]
    print(Mb_counts)
    Mb_count = Counter(Mb_counts).most_common(1)[0][0]
    # Compute winding numbers
    Wdeg1 = ((Mdeg1_count - Mb_count) % 2 + 1) % 2
    Wdeg2 = ((Mdeg2_count - Mb_count) % 2 + 1) % 2
    Wtotal = Wdeg1 + Wdeg2
    return {
        'Wtotal': Wtotal,
        'Wdeg1': Wdeg1,
        'Wdeg2': Wdeg2,
        'Mdeg1_count': Mdeg1_count,
        'Mdeg2_count': Mdeg2_count,
        'Mb_count': Mb_count,
        'Mdeg1': Mdeg1,
        'Mdeg2': Mdeg2,
        'interp_curves': (
            interp_plus,
            interp_minus
        )
    }


def create_main_figure_H(eigenvalues, eigenvectors, Eedge, Enotedge,
                         zbz, zbranchpts,
                         mbzplus, mbzminus, edge_indicators,
                         left_edge_indicator, right_edge_indicator,
                         Mdeg1, Mdeg2,
                         interp_real_plus_closed, interp_imag_plus_closed,
                         interp_real_minus_closed, interp_imag_minus_closed,
                         t, var, is_hermitian, Wbz1, Wbz2):
    cmap = plt.cm.hsv
    norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)

    plt.figure(figsize=(12, 20))

    # 1. Eigenvector localization
    plt.subplot(4, 2, 1)
    plt.imshow(np.abs(eigenvectors), cmap='inferno', aspect='auto')
    plt.colorbar(label=r'$|\psi_n|$')
    plt.xlabel("$\mathrm{eigenvector\ index}$")
    plt.ylabel("$\mathrm{site}$")
    plt.title(f"$t={t:.3g},\; \\alpha={var:.3g}$")

    # 2. Eigenvalue spectrum
    plt.subplot(4, 2, 2)
    plt.plot(np.real(Eedge), np.imag(Eedge), 'o',
             markersize=10, markeredgecolor='g', markerfacecolor='none', markeredgewidth=3,
             label='$E_{\\mathrm{edge}}$')
    plt.plot(np.real(Enotedge), np.imag(Enotedge), 'o',
             markersize=10, markeredgecolor='r', markerfacecolor='none', markeredgewidth=3,
             label='$E_{\\mathrm{bulk}}$')
    plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'o', markersize=3, color='black')
    plt.xlabel("$\\mathrm{Re}(E)$")
    plt.ylabel("$\\mathrm{Im}(E)$")
    plt.title(f"$\\mathrm{{analytical\;edges}}={len(Eedge)}$", pad=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), frameon=False, ncol=2)
    if is_hermitian:
        plt.ylim([-1, 1])

    # 3. zbz on z plane
    plt.subplot(4, 2, 3)
    ax = plt.gca()
    plt.scatter([z.real for z in zbranchpts], [z.imag for z in zbranchpts], color='green', label='$z_{\\mathrm{branch}}$')
    plt.scatter([z.real for z in zbz], [z.imag for z in zbz],
                c=np.angle(zbz), cmap=cmap, norm=norm, marker='.')
    plt.xlabel('$\\mathrm{Re}(z)$')
    plt.ylabel('$\\mathrm{Im}(z)$')
    plt.title('$z_{\\mathrm{bz}}^{(1)}$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')
    plt.legend(bbox_to_anchor=(0.5, 1.30), loc='upper center', ncol=2,
            frameon=False,
            columnspacing=0.5,
            handletextpad=0.1)

    # 4. zbz on z plane
    plt.subplot(4, 2, 4)
    ax = plt.gca()
    plt.scatter([z.real for z in zbranchpts], [z.imag for z in zbranchpts], color='green', label='$z_{\\mathrm{branch}}$')
    plt.scatter([z.real for z in zbz], [z.imag for z in zbz],
                c=np.angle(zbz), cmap=cmap, norm=norm, marker='.')
    plt.xlabel('$\\mathrm{Re}(z)$')
    plt.ylabel('$\\mathrm{Im}(z)$')
    plt.title('$z_{\\mathrm{bz}}^{(2)}$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')
    plt.legend(bbox_to_anchor=(0.5, 1.30), loc='upper center', ncol=2,
            frameon=False,
            columnspacing=0.5,
            handletextpad=0.1)
    
    # 5. edge indicator (3D)
    plt.subplot(4, 2, 5, projection='3d')
    ax = plt.gca()
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), edge_indicators, c='black')
    ax.set_xlabel('$\\mathrm{Re}(E)$', labelpad=10)
    ax.set_ylabel('$\\mathrm{Im}(E)$', labelpad=10)
    ax.set_zlabel('$\\mathcal{I}_{\\mathrm{edge}}$', labelpad=10)
    ax.set_title('$\\mathrm{edge\\ indicator}$', pad=20)
    if is_hermitian:
        ax.set_ylim([-1, 1])
    max_z = np.max(right_edge_indicator)
    min_z = np.min(left_edge_indicator)
    ax.set_zlim([min(1.1 * min_z, -1), max(1.1 * max_z, 1)])

    # 6. Edge indicator vs index
    plt.subplot(4, 2, 6)
    plt.scatter(range(len(edge_indicators)), edge_indicators, c='black')
    plt.xlabel('$\\mathrm{Eig\\;index}$')
    plt.ylabel('$\\mathcal{I}_{\\mathrm{edge}}$')
    plt.title(
        f"$\\mathrm{{left\\;edge\\;indicator}}:{left_edge_indicator:.2f}$\n"
        f"$\\mathrm{{right\\;edge\\;indicator}}:{right_edge_indicator:.2f}$",
        pad=25
    )
    plt.ylim([min(1.1 * min_z, -1), max(1.1 * max_z, 1)])

    # 7. M-plane: Wbz1
    plt.subplot(4, 2, 7)
    ax = plt.gca()
    plt.scatter(Mdeg1.real, Mdeg1.imag, color='blue', label='$M_{\\mathrm{deg}}^{(1)}$')
    plt.scatter(Mdeg2.real, Mdeg2.imag, color='blue', label='$M_{\\mathrm{deg}}^{(2)}$')
    plt.plot(interp_real_plus_closed, interp_imag_plus_closed, color='blue', linestyle='-', alpha=0.3)
    plt.scatter([m.real for m in mbzplus], [m.imag for m in mbzplus],
                c=np.angle(zbz), cmap=cmap, norm=norm, marker='.')
    plt.xlabel(r'$\mathrm{Re}(M)$')
    plt.ylabel(r'$\mathrm{Im}(M)$')
    plt.title(r'$W_{\mathrm{bz}}^{(1)}=' + f'{Wbz1}' + '$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')

    # 8. M-plane: Wbz2
    plt.subplot(4, 2, 8)
    ax = plt.gca()
    plt.scatter(Mdeg1.real, Mdeg1.imag, color='blue', label='$M_{\\mathrm{deg}}^{(1)}$')
    plt.scatter(Mdeg2.real, Mdeg2.imag, color='blue')
    plt.plot(interp_real_minus_closed, interp_imag_minus_closed, color='blue', linestyle='-', alpha=0.3)
    plt.scatter([m.real for m in mbzminus], [m.imag for m in mbzminus],
                c=np.angle(zbz), cmap=cmap, norm=norm, marker='.')
    plt.xlabel(r'$\mathrm{Re}(M)$')
    plt.ylabel(r'$\mathrm{Im}(M)$')
    plt.title(r'$W_{\mathrm{bz}}^{(2)}=' + f'{Wbz2}' + '$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')

    plt.subplots_adjust(left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.7, 
            hspace=0.9) 


def create_main_figure_NH(eigenvalues, eigenvectors, Eedge, Enotedge,
                          zgbzplus, zgbzminus, zbranchpts, 
                          mgbzplus, mgbzminus, edge_indicators,
                          left_edge_indicator, right_edge_indicator, Mbranchlist,
                          Mdeg1, Mdeg2, 
                          interp_real_plus, interp_imag_plus, interp_real_minus, interp_imag_minus,
                          t, var, is_hermitian, Wdeg1, Wdeg2, Mdeg1_count, Mdeg2_count, Mb_count):

    cmap = plt.cm.hsv
    norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)

    plt.figure(figsize=(12, 20))

    # 1. Eigenvector localization
    plt.subplot(4, 2, 1)
    plt.imshow(np.abs(eigenvectors), cmap='inferno', aspect='auto')
    plt.colorbar(label=r'$|\psi_n|$')
    plt.xlabel("$\mathrm{eigenvector\ index}$")
    plt.ylabel("$\mathrm{site}$")
    plt.title(f"$t={t:.3g},\; \\alpha={var:.3g}$")

    # 2. Eigenvalue spectrum
    plt.subplot(4, 2, 2)
    plt.plot(np.real(Eedge), np.imag(Eedge), 'o',
             markersize=10, markeredgecolor='g', markerfacecolor='none', markeredgewidth=3,
             label='$E_{\\mathrm{edge}}$')
    plt.plot(np.real(Enotedge), np.imag(Enotedge), 'o',
             markersize=10, markeredgecolor='r', markerfacecolor='none', markeredgewidth=3,
             label='$E_{\\mathrm{bulk}}$')
    plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'o', markersize=3, color='black')
    plt.xlabel("$\\mathrm{Re}(E)$")
    plt.ylabel("$\\mathrm{Im}(E)$")
    plt.title(f"$\\mathrm{{analytical\;edges}}={len(Eedge)}$", pad=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), frameon=False, ncol=2)

    # 3. zGBZ+
    plt.subplot(4, 2, 3)
    ax = plt.gca()
    plt.scatter([z.real for z in zbranchpts], [z.imag for z in zbranchpts],
                color='green', label='$z_{\\mathrm{branch}}$')
    plt.scatter([z.real for z in zgbzplus], [z.imag for z in zgbzplus],
                c=np.angle(zgbzplus), cmap=cmap, norm=norm, marker='.')
    plt.xlabel('$\\mathrm{Re}(z)$')
    plt.ylabel('$\\mathrm{Im}(z)$')
    plt.title('$z_{\\mathrm{gbz}}^{(1)}$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')
    plt.legend(bbox_to_anchor=(0.5, 1.30), loc='upper center', ncol=2,
            frameon=False,
            columnspacing=0.5,
            handletextpad=0.1)

    # 4. zGBZ-
    plt.subplot(4, 2, 4)
    ax = plt.gca()
    plt.scatter([z.real for z in zbranchpts], [z.imag for z in zbranchpts],
                color='green', label='$z_{\\mathrm{branch}}$')
    plt.scatter([z.real for z in zgbzminus], [z.imag for z in zgbzminus],
                c=np.angle(zgbzminus), cmap=cmap, norm=norm, marker='.')
    plt.xlabel('$\\mathrm{Re}(z)$')
    plt.ylabel('$\\mathrm{Im}(z)$')
    plt.title('$z_{\\mathrm{gbz}}^{(2)}$', pad=45)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')
    plt.legend(bbox_to_anchor=(0.5, 1.30), loc='upper center', ncol=2,
            frameon=False,
            columnspacing=0.5,
            handletextpad=0.1)

    # 5. Edge indicator 3D
    plt.subplot(4, 2, 5, projection='3d')
    ax = plt.gca()
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), edge_indicators, c='black')
    ax.set_xlabel('$\\mathrm{Re}(E)$', labelpad=10)
    ax.set_ylabel('$\\mathrm{Im}(E)$', labelpad=10)
    ax.set_zlabel('$\\mathcal{I}_{\\mathrm{edge}}$', labelpad=10)
    ax.set_title('$\\mathrm{edge\\ indicator}$', pad=20)
    if is_hermitian:
        ax.set_ylim([-1, 1])
    max_z = np.max(right_edge_indicator)
    min_z = np.min(left_edge_indicator)
    ax.set_zlim([min(1.1 * min_z, -1), max(1.1 * max_z, 1)])

    # 6. Edge indicator vs index
    plt.subplot(4, 2, 6)
    plt.scatter(range(len(edge_indicators)), edge_indicators, c='black')
    plt.xlabel('$\\mathrm{Eig\\;index}$')
    plt.ylabel('$\\mathcal{I}_{\\mathrm{edge}}$')
    plt.title(
        f"$\\mathrm{{left\\;edge\\;indicator}}:{left_edge_indicator:.2f}$\n"
        f"$\\mathrm{{right\\;edge\\;indicator}}:{right_edge_indicator:.2f}$",
        pad=25
    )
    plt.ylim([min(1.1 * min_z, -1), max(1.1 * max_z, 1)])

    # 7. M-plane with GBZ curves and branch points (Wdeg1)
    plt.subplot(4, 2, 7)
    ax = plt.gca()
    plt.scatter(Mdeg1.real, Mdeg1.imag, color='blue', label='$M_{\\mathrm{deg}}^{(1)}$')
    plt.scatter(Mbranchlist.real, Mbranchlist.imag, color='red', label='$M_{\\mathrm{branch}}$')
    plt.plot(interp_real_plus, interp_imag_plus, color='blue', linestyle='-', alpha=0.3)
    plt.plot(interp_real_minus, interp_imag_minus, color='blue', linestyle='-', alpha=0.3)
    plt.scatter([m.real for m in mgbzplus], [m.imag for m in mgbzplus],
                c=np.angle(zgbzplus), cmap=cmap, norm=norm, marker='.')
    plt.scatter([m.real for m in mgbzminus], [m.imag for m in mgbzminus],
                c=np.angle(zgbzminus), cmap=cmap, norm=norm, marker='.')

    plt.xlabel(r'$\mathrm{Re}(M)$')
    plt.ylabel(r'$\mathrm{Im}(M)$')
    title_str = (
        r'$W_{1}=\mathrm{mod}_{2}(1+W_{\mathrm{deg,1}}-W_{\mathrm{branch}})$' + '\n' +
        r'$=\mathrm{mod}_{2}(1+' + f'{Mdeg1_count}-{Mb_count}' + ')=' + f'{Wdeg1}' + '$'
    )
    plt.title(title_str, pad=40)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')


    # 8. M-plane with GBZ curves and branch points (Wdeg2)
    plt.subplot(4, 2, 8)
    ax = plt.gca()
    plt.scatter(Mdeg2.real, Mdeg2.imag, color='blue', label='$M_{\\mathrm{deg}}^{(2)}$')
    plt.scatter(Mbranchlist.real, Mbranchlist.imag, color='red', label='$M_{\\mathrm{branch}}$')
    plt.plot(interp_real_plus, interp_imag_plus, color='blue', linestyle='-', alpha=0.3)
    plt.plot(interp_real_minus, interp_imag_minus, color='blue', linestyle='-', alpha=0.3)
    plt.scatter([m.real for m in mgbzplus], [m.imag for m in mgbzplus],
                c=np.angle(zgbzplus), cmap=cmap, norm=norm, marker='.')
    plt.scatter([m.real for m in mgbzminus], [m.imag for m in mgbzminus],
                c=np.angle(zgbzminus), cmap=cmap, norm=norm, marker='.')

    plt.xlabel(r'$\mathrm{Re}(M)$')
    plt.ylabel(r'$\mathrm{Im}(M)$')
    title_str = (
        r'$W_{2}=\mathrm{mod}_{2}(1+W_{\mathrm{deg,2}}-W_{\mathrm{branch}})$' + '\n' +
        r'$=\mathrm{mod}_{2}(1+' + f'{Mdeg2_count}-{Mb_count}' + ')=' + f'{Wdeg2}' + '$'
    )
    plt.title(title_str, pad=40)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='$\\arg(z)$')

    plt.subplots_adjust(left=0.1,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.7, 
            hspace=0.9) 