from tight_binding.model import TightBinding
from tight_binding.objects import UnitCell
from tight_binding_parameters import ParameterCollection
from sympy.vector import CoordSys3D
from pprint import pprint
import matplotlib.pyplot as plt
from fitting import Fitting_data
import argparse
import numpy as np


def parse_arguments() -> argparse.Namespace:  # Handles the input parameters from the command line.
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Generate a bandstructure using a Tight Binding model for a certain material""")
    parser.add_argument(
        "material", help="Material for which to generate a bandstructure")
    parser.add_argument("order", type=int,
                        help="Maximum order of nearest neighbours that are included. Only used when wannier = 0")
    parser.add_argument("--symmetry_points", metavar="symmetry-point", nargs="*", type=str,
                        help="Symmetry points through which to generate bandstructures")
    parser.add_argument("--wannier", type=int,
                        help="Specify whether you want to use matrices from wannier90")
    parser.add_argument("--nk", type=int,
                        help="Number of points in each direction of k. Grid size increases with the value cubed")
    parser.add_argument("--Ef", type=float,
                        help="Roughly specify the location of the Fermi level. Used to determine which bands are "
                             "valence and conduction bands. Use this when the top of the valence band is "
                             "not at 0. Default value is 0.", default=0)
    parser.add_argument("--adaptive_grid_length", type=int, default=1,
                        help="Adaptive grid length determines the number of points between symmetry points based on "
                             "the distance between the respective symmetry points.")
    parser.add_argument("--data_file_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the data file")
    parser.add_argument("--unit_cell_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the unit file")
    args = parser.parse_args()
    return args


def find_conduction_band(eigenvalues):
    # Find the location of the conduction bands. This is needed when when working in a new basis (the eigenvectors).
    iscdband = np.zeros(len(eigenvalues[:, 0]))
    for index, eigenvalue_row in enumerate(eigenvalues):  # It is sufficient to look at the first values only since
        # all of the values inside the band are either positive or negative.
        if np.average(eigenvalue_row) >= 0:  # if true, most if not all of the band is above the Fermi level
            iscdband[index] = 1
    return iscdband


def transition_matrix_elements(eigenvalues, eigenvectors, iscdband, dH, spin_orbit_coupling):
    # TODO: Finish the function to calculate the (dipole) transition matrix elements. equation 42 of report
    eigenvalues_c = eigenvalues[np.nonzero(np.round(iscdband))[0], :]
    eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]

    eigenvectors_c = eigenvectors[:, :, np.nonzero(np.round(iscdband))[0]]  # conduction band eigenvectors
    eigenvectors_v = eigenvectors[:, :, np.nonzero(np.round(1 - iscdband))[0]]  # valence band eigenvectors

    if spin_orbit_coupling:
        tme = np.zeros(np.shape(eigenvalues)[1])
        for i in range(0, 2):
            for j in range(0, 2):
                eigs_c = eigenvectors_c[:, :, i].T
                eigs_v = eigenvectors_v[:, :, -1 - j].T
                c_cv = np.conjugate(eigs_c[:, np.newaxis]) * eigs_v
                P_cv = np.sum(c_cv[..., np.newaxis] * dH[:, :, :], axis=(0, 1))
                tme += (np.sum(np.real(np.conjugate(P_cv) * P_cv), axis=1) / 3 / 4) ** 2
    else:
        eigs_c = eigenvectors_c[:, :, 0].T  # CBM
        eigs_v = eigenvectors_v[:, :, -1].T  # VBM
        c_cv = np.conjugate(eigs_c[:, np.newaxis]) * eigs_v
        P_cv = np.sum(c_cv[..., np.newaxis] * dH[:, :, :], axis=(0, 1))
        tme = np.sum(np.real(np.conjugate(P_cv) * P_cv), axis=1) / 3
    return tme


if __name__ == "__main__":
    args = parse_arguments()
    nk = args.nk
    data_file = args.material + args.data_file_suffix
    r = CoordSys3D("r")
    unit_cell = UnitCell.from_file("./unit_cells/{}.json".format(args.material + args.unit_cell_suffix), r)
    model = TightBinding(unit_cell, r, args.order)
    k_values, n_pos_symmetry_points = unit_cell.get_k_values(args.nk, args.symmetry_points,
                                                             adaptive=args.adaptive_grid_length)
    if args.wannier == 1:
        data = Fitting_data(data_file)
        input_matrices_R = data.matrices
        matrix_shape = np.shape(next(iter(input_matrices_R.items()))[1])
        input_matrices_k, dH, eigenvalues, eigenvectors = model.matrices_R_to_k(
            matrix_shape, input_matrices_R, k_values, r)
    else:
        matrix = model.construct_hamiltonian()
        pprint(matrix)
        parameters_collection = ParameterCollection.from_file(
            "tight_binding_parameters/{}.json".format(data_file), str(args.order))
        eigenvalues, eigenvectors = model.get_energy_eigen({**parameters_collection.energies,
                                                            **parameters_collection.overlap_integrals,
                                                            **parameters_collection.soc_coefficients},
                                                           k_values)
        import dielectric_function

        dH = dielectric_function.differentiate_hamiltonian(matrix, parameters_collection, model.k_symbols, k_values)

    """calculating tme"""

    iscdband = find_conduction_band(eigenvalues - args.Ef)
    eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]
    shift = np.max(eigenvalues_v)
    eigenvalues = eigenvalues - shift
    print('Number of valence bands: ' + str(int(np.sum(-(iscdband - 1)))))
    print('Number of conduction bands: ' + str(int(np.sum(iscdband))))

    tme = transition_matrix_elements(eigenvalues, eigenvectors, iscdband, dH, unit_cell.spin_orbit_coupling)

    import data_processing

    parameter_keys = ['nk', "n_pos_symmetry_points", "labels"]
    n_pos_symmetry_points = [int(x) for x in n_pos_symmetry_points]
    parameter_values = [args.nk, n_pos_symmetry_points, args.symmetry_points]
    data_keys = ['Bandstructures', 'tme']
    data_values = [eigenvalues, tme]
    data = data_processing.to_dict(args.material, data_keys, data_values, parameter_keys, parameter_values)
    data_processing.save_as_json(data)

    plt.figure(1)
    plt.xticks(n_pos_symmetry_points, [f"${symmetry_point}$" for symmetry_point in args.symmetry_points])
    plt.title('Transition matrix elements: ' + args.material)
    plt.plot(tme)
    plt.show()
