from tight_binding.model import TightBinding
from tight_binding.objects import UnitCell
from tight_binding_parameters import ParameterCollection
from sympy.vector import CoordSys3D
from pprint import pprint
from sympy import diff, symbols, lambdify
import itertools
from math import sqrt
import numpy.linalg as LA
from fitting import Fitting_data
import psutil
import data_processing
import argparse
import numpy as np
import time

m0 = 9.10938e-31
hbar = 1.05457e-34
epsilon_0 = 8.8541878128e-12
q = 1.602176634e-19
# n_ref = 2.4293  # index of refraction
c = 299792458
A0 = 1


def parse_arguments() -> argparse.Namespace:  # Handles the input parameters from the command line.
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Calculate optical properties using the eigenvalues, eigenvectors and matrix
                        elements.""")
    parser.add_argument("material",
                        help="Material for which to generate a bandstructure")
    parser.add_argument("--order", type=int,
                        help="Maximum order of nearest neighbours that are included. Only used when wannier = 0")
    parser.add_argument("--nk", type=int,
                        help="Number of points in each direction of k. Grid size increases with the value cubed")
    parser.add_argument("--N", type=int,
                        help="Number of points used for the energy range of the photon")
    parser.add_argument("--Ep", type=float,
                        help="Range for the energy of the photon")
    parser.add_argument("--direction", type=int, default=None, nargs=3,
                        help="""Direction in which we calculate the dielectric function. 3 inputs required for 
                             the component of each of the 3 directions""")
    parser.add_argument("--wannier", type=int, default=0,
                        help="Specify whether you want to use matrices from wannier90")
    parser.add_argument("--Ef", type=float, default=0,
                        help="""Roughly specify the location of the Fermi level. Used to determine which bands are
                             valence and conduction bands. Use this when the top of the valence band is 
                             not at 0. Default value is 0.""")
    parser.add_argument("--nv", type=int, default=np.inf,
                        help="""The number of valence bands included in the calculations. By default we take all valence 
                             bands""")
    parser.add_argument("--nc", type=int, default=np.inf,
                        help="""The number of conduction bands included in the calculations. By default we take all 
                             conduction bands""")
    parser.add_argument("--fwhm", type=float, default=0.1,
                        help="The full width half maximum of the gaussian")
    parser.add_argument("--symmetry_points", metavar="symmetry-point", nargs="*", type=str,
                        help="Symmetry points through which to generate bandstructures")
    parser.add_argument("--data_file_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the data file")
    parser.add_argument("--unit_cell_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the unit file")
    parser.add_argument("--compare", type=int, default=0,
                        help="After the calculations are done this allows you to select a JSON file which we will"
                             "plot together with the results")
    parser.add_argument("--save", type=int, default=1,
                        help="Indicate whether or not the save the results. True on default.")
    args = parser.parse_args()
    if args.direction:
        args.direction = [element / np.linalg.norm(args.direction) for element in args.direction]  # normalization
    return args


def substitute(parameters, k_symbols, k_values, matrix):
    """Susbtitute parameter values and k values inside the symbolic Hamiltonian"""
    substitutions = {}
    for symbol in matrix.free_symbols.difference(k_symbols):  # Determine which symbols to substitute
        substitutions[symbol] = parameters[str(symbol)]

    z = symbols("z")
    for i, j in itertools.product(range(matrix.shape[0]), repeat=2):
        if matrix[i, j] == 0:
            matrix[i, j] = z  # Replaces zero elements with z

    variables = list(itertools.chain(substitutions.keys(), ['kx', 'ky', 'kz'], [z]))
    lambda_matrix = lambdify(variables, matrix)  # Set the variables in the matrix

    params = np.tile(np.atleast_2d(list(substitutions.values())).T, (1, k_values.shape[1]))
    zeros = np.zeros(k_values.shape[1])

    return lambda_matrix(*params, *k_values, zeros)


def differentiate_hamiltonian(hamiltonian, parameters_collection, k_symbols, k_values):
    """Symbolically calculate the gradient of the Hamiltonian and then substitute parameter values and k values."""
    nabla_kx_hamiltonian = diff(hamiltonian, k_symbols[0])
    nabla_ky_hamiltonian = diff(hamiltonian, k_symbols[1])
    nabla_kz_hamiltonian = diff(hamiltonian, k_symbols[2])
    dx_hamiltonian = substitute({**parameters_collection.energies, **parameters_collection.overlap_integrals},
                                k_symbols, k_values, nabla_kx_hamiltonian)
    dy_hamiltonian = substitute({**parameters_collection.energies, **parameters_collection.overlap_integrals},
                                k_symbols, k_values, nabla_ky_hamiltonian)
    dz_hamiltonian = substitute({**parameters_collection.energies, **parameters_collection.overlap_integrals},
                                k_symbols, k_values, nabla_kz_hamiltonian)
    dH = np.array([dx_hamiltonian, dy_hamiltonian, dz_hamiltonian])
    dH = np.moveaxis(dH, 0, -1)
    return dH


def delta(energy_conduction, energy_valence, energy_photon, tolerance):
    if abs(energy_conduction - energy_valence - energy_photon) <= tolerance / 2:
        value = 1
    else:
        value = 0
    return value


def gaussian_broadening(energy, full_width_half_maximum):
    """The gaussian is centered around 0. So you should subtract the photon energy before passing it to this function.
    When the energy is at half the value of the full width half maximum (the half width half maximum) this function will
    return half of the peak value."""
    gaussian = 2 / full_width_half_maximum * np.sqrt(np.log(2) / np.pi) * np.exp(
        -4 * np.log(2) * (energy / full_width_half_maximum) ** 2)
    return gaussian


def calculate_dielectric_function(eigenvalues, eigenvectors, iscdband, dH, direction, N_Ep, Ep_scope, fwhm,
                                  indices_per_loop=None):
    """Function that calculates the joint density of states, the real and imaginary part of the dielectric function,
    the index of refraction, and the absorption.
    indices_per_loop determines the amount of indices that are calculated at once in a vectorized way. If not given
    we will take a value dependent on the amount of available memory."""
    eigenvalues_c = eigenvalues[np.nonzero(np.round(iscdband))[0], :]
    eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]
    eig_difference = np.zeros([eigenvalues_c.shape[0], eigenvalues_v.shape[0], eigenvalues.shape[1]])
    for n_c in range(len(eigenvalues_c)):
        for n_v in range(len(eigenvalues_v)):
            eig_difference[n_c, n_v, :] = eigenvalues_c[n_c] - eigenvalues_v[n_v]  # contains all possible transitions
    tolerance = 2 * fwhm
    Ep_range = np.linspace(0, Ep_scope, N_Ep)
    eigenvectors_c = eigenvectors[:, :, np.nonzero(np.round(iscdband))[0]]  # conduction band eigenvectors
    eigenvectors_v = eigenvectors[:, :, np.nonzero(np.round(1 - iscdband))[0]]  # valence band eigenvectors
    epsilon_i = np.zeros(np.shape(Ep_range))
    joint_DOS = np.zeros(np.shape(Ep_range))

    dH_size = np.shape(dH)[0] * np.shape(dH)[1] * np.shape(dH)[3]

    if direction:
        e_dH = np.sum(direction * dH, axis=3)  # Get the inner product between dH and the direction vector
    else:
        e_dH = None  # Will not be used in this case

    for index_p, Ep in enumerate(Ep_range):  # Loop over the photon energies
        eig_true = ((eig_difference >= (Ep - tolerance)) == (eig_difference <= (Ep + tolerance)))
        # eig_true checks which transitions are within a certain tolerance such that the gaussian will give nonzero.
        all_indices = np.array(eig_true.nonzero())
        # all_indices gives the indices for the conduction band, valence bands, and k_values for which eig_true is true.
        length = np.shape(all_indices)[1]  # amount of indices which we need to sum over
        memory_usage = psutil.virtual_memory()
        if indices_per_loop is None:
            indices_per_loop = (memory_usage.available / (16 * dH_size * 3))
        n = int(np.ceil(length / indices_per_loop))
        # n indicates the amount of times we run the loop. The factor 16 is the size of one complex array element.
        # The factor 3 is to make sure we have enough memory to run the calculations (might need to change if needed).
        print('memory available: ' + str(memory_usage.available / 1E9) + ' (gB). ' + str(length) +
              ' possible transitions with Ep = ' + str('{0:.3f}'.format(Ep)) + ', solving in ' + str(n) +
              ' iteration(s)')
        if n > 0:  # only do the calculations if there are transitions possible for the given photon energy
            """ We use a hybridized way to calculate the dielectric function: the vectorization is split into parts. 
                This reduces the memory usage which can otherwise be too large to be able to run this code. If you have 
                memory issues consider lowering indices_per_loop."""
            for indices in np.array_split(all_indices, n, axis=1):
                # we split the calculations in parts to prevent too much memory usage
                indices_c, indices_v, indices_k = indices[0], indices[1], indices[2]
                eigs_c = eigenvectors_c[indices_k, :, indices_c].T  # eigenvalues for the conduction bands
                eigs_v = eigenvectors_v[indices_k, :, indices_v].T  # eigenvalues for the valence bands
                c_cv = np.conjugate(eigs_c[:, np.newaxis]) * eigs_v
                # c_cv is a matrix containing the multiplication between each eigenvector element with the same k-value.
                # the 3rd dimension which we generate is for the k values
                if direction:  # if true: Calculate over a predefined direction
                    P_cv = np.sum(c_cv * e_dH[:, :, indices_k], axis=(0, 1))  # transition matrix elements (non squared)
                    epsilon_i[index_p] += np.sum(1 / Ep ** 2 * np.real(np.conjugate(P_cv) * P_cv) * gaussian_broadening(
                        eigenvalues_c[indices_c, indices_k] - eigenvalues_v[indices_v, indices_k] - Ep, fwhm))
                else:  # else: Take an average over all three principal directions
                    P_cv = np.sum(c_cv[..., np.newaxis] * dH[:, :, indices_k], axis=(0, 1))
                    # in this case we need to add an extra dimension to c_cv for the three principal directions
                    epsilon_i[index_p] += np.sum(1 / Ep ** 2 * np.sum(np.real(np.conjugate(P_cv) * P_cv), axis=1) / 3 *
                                                 gaussian_broadening(eigenvalues_c[indices_c, indices_k] -
                                                                     eigenvalues_v[indices_v, indices_k] - Ep, fwhm))
                joint_DOS[index_p] += np.sum(gaussian_broadening(
                    eigenvalues_c[indices_c, indices_k] - eigenvalues_v[indices_v, indices_k] - Ep, fwhm))
            del c_cv, P_cv, indices, indices_k, indices_c, indices_v, eigs_c, eigs_v
        print("Done with: Ep = " + str('{0:.3f}'.format(Ep)) + ", progress: " + str(index_p + 1) + '/' + str(N_Ep))
    epsilon_i = 8 * np.pi ** 2 * epsilon_i / eigenvalues.shape[1]
    joint_DOS = joint_DOS / eigenvalues.shape[1]

    epsilon_r = np.ones(np.shape(Ep_range))  # we start with ones since it is in the formula for epsilon_r as a constant
    dE2 = Ep_scope / (N_Ep - 1)  # the steps in the frequency for the Riemann's sum
    gradient_epsilon_i = np.gradient(epsilon_i, Ep_range)
    for index_p, E in enumerate(Ep_range):  # Loop for epsilon_r, loop over omega
        for index_p2, E2 in enumerate(Ep_range):  # Loop for epsilon_r, loop over omega'
            if E != E2:  # energies can't be the same since the equation is indeterminate (0/0)
                epsilon_r[index_p] += 2 / np.pi * (E2 * epsilon_i[index_p2] - E * epsilon_i[index_p]) / (
                        E2 ** 2 - E ** 2) * dE2
            else:  # evaluating the limit E2 -> E using l'Hopital's rule to avoid an indeterminate expression
                epsilon_r[index_p] += 2 / np.pi * ((epsilon_i[index_p2] + E2 * gradient_epsilon_i[index_p2] -
                                                    E * epsilon_i[index_p]) / (2 * E2 - E ** 2)) * dE2

    n_ref = np.zeros(np.shape(Ep_range))  # index of refraction
    for index_p, Ep in enumerate(Ep_range):
        n_ref[index_p] = 1 / 2 * (epsilon_r[index_p] + sqrt(epsilon_r[index_p] ** 2 + epsilon_i[index_p] ** 2))
    # n_ref = np.ones(np.shape(Ep_range)) * 2.4293
    absorption = (Ep_range * q / hbar) / (c * n_ref) * epsilon_i * 0.01  # in cm^-1

    return epsilon_i, epsilon_r, absorption, n_ref, Ep_range, joint_DOS


def main():
    time1 = time.perf_counter()
    args = parse_arguments()
    nk = args.nk
    N_Ep = args.N
    Ep_scope = args.Ep
    fwhm = args.fwhm

    args = parse_arguments()

    r = CoordSys3D("r")
    unit_cell = UnitCell.from_file("./unit_cells/{}.json".format(args.material + args.unit_cell_suffix), r)
    model = TightBinding(unit_cell, r, args.order)
    k_values = unit_cell.k_values_ibz(nk)
    data_file = args.material + args.data_file_suffix
    if args.wannier:
        data = Fitting_data('matrices/' + data_file)
        input_matrices_R = data.matrices
        input_matrices_k, dH, eigenvalues, eigenvectors = model.matrices_R_to_k(input_matrices_R, k_values, r)
    else:
        symHamiltonian = model.construct_hamiltonian()
        pprint(symHamiltonian)

        # Get the parameter values from a JSON file:
        parameters_collection = ParameterCollection.from_file(
            "tight_binding_parameters/{}.json".format(data_file), str(args.order))

        eigenvalues, eigenvectors = model.get_energy_eigen(
            {**parameters_collection.energies, **parameters_collection.overlap_integrals,
             **parameters_collection.soc_coefficients}, k_values)
        dH = differentiate_hamiltonian(symHamiltonian, parameters_collection, model.k_symbols, k_values)

    iscdband = model.find_conduction_band(eigenvalues - args.Ef)
    eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]
    shift = np.max(eigenvalues_v)
    eigenvalues = eigenvalues - shift  # set the top of the valence band to zero
    eigenvalues_c = eigenvalues[np.nonzero(np.round(iscdband))[0], :]
    eigenvalues_v = eigenvalues_v - shift

    valence_count = int(np.sum(-(iscdband - 1)))
    conductance_count = int(np.sum(iscdband))
    print('Number of valence bands: ' + str(valence_count))
    print('Number of conduction bands: ' + str(conductance_count))

    chosen_bands = slice(valence_count - args.nv if valence_count - args.nv > 0 else 0,
                         valence_count + args.nc if valence_count + args.nc < valence_count + conductance_count else
                         valence_count + conductance_count)  # only look at these bands during the calculations
    epsilon_i, epsilon_r, absorption, n_ref, Ep, JDOS = calculate_dielectric_function(
        eigenvalues[chosen_bands, :], eigenvectors[:, :, chosen_bands], iscdband[chosen_bands],
        dH, args.direction, N_Ep, Ep_scope, fwhm)

    time2 = time.perf_counter()
    elapsed_time = time2 - time1
    print("Elapsed time (dielectric function calculation): " + str(elapsed_time))

    parameter_keys = ['nk', 'N_Ep', 'Ep_scope', 'fwhm', 'Ef', 'nv', 'nc', 'direction', 'data_file_suffix',
                      'unit_cell_suffix']
    parameter_values = [nk, N_Ep, Ep_scope, fwhm, args.Ef, args.nv, args.nc, args.direction,
                        args.data_file_suffix, args.unit_cell_suffix]
    data_keys = ['elapsed_time', 'Ep_range', 'Eg', 'E_max', 'E_min', '\u03B1', '\u03B5_i', '\u03B5_r', 'n_ref',
                 'JDOS']  # epsilon_i = '\u03B5_i', alpha = '\u03B1'
    data_values = [elapsed_time, Ep, np.min(eigenvalues_c) - np.max(eigenvalues_v), np.max(eigenvalues),
                   np.min(eigenvalues), absorption, epsilon_i, epsilon_r, n_ref, JDOS]
    data = data_processing.to_dict(args.material, data_keys, data_values, parameter_keys, parameter_values)
    if args.save == 1:
        data_processing.save_as_json(data)
    if args.compare:
        data_processing.generate_plots(data_list=[data], number_of_plots=2, labels=['model', 'external'])
    else:
        data_processing.generate_plots(data_list=[data], number_of_plots=1)

    return


if __name__ == "__main__":
    main()
