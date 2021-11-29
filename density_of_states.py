from tight_binding.model import TightBinding
from tight_binding.objects import UnitCell
from tight_binding_parameters import ParameterCollection
import gen_bandstructure
from sympy.vector import CoordSys3D
from pprint import pprint
import matplotlib.pyplot as plt
from fitting import Fitting_data
import data_processing

import argparse
import numpy as np
import time


def parse_arguments() -> argparse.Namespace:  # Handles the input parameters from the command line.
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Generate a bandstructure using a Tight Binding model for a certain material""")
    parser.add_argument("material",
                        help="Material for which to generate a bandstructure")
    parser.add_argument("order", type=int,
                        help="Maximum order of nearest neighbours that are included. Only used when wannier = 0")
    parser.add_argument("--symmetry_points", metavar="symmetry-point", nargs="*", type=str,
                        help="Symmetry points through which to generate bandstructures")
    parser.add_argument("--wannier", type=int,
                        help="Specify whether you want to use matrices from wannier90")
    parser.add_argument("--adaptive_grid_length", type=int, default=1,
                        help="Adaptive grid length determines the number of points between symmetry points based "
                             "on the distance between the respective symmetry points.")
    parser.add_argument("--nk", type=int, default=50,
                        help="Length of each dimension for the k-space grid. Length of the k-values array scales with "
                             "this value cubed.")
    parser.add_argument("--N", type=int, default=1000,
                        help="amount of steps in the energy for calculating the DOS")
    parser.add_argument("--fwhm", type=float, default=0.1,
                        help="The full width half maximum of the gaussian")
    parser.add_argument("--Ef", type=float, default=0,
                        help="Roughly specify the location of the Fermi level. Used to determine which bands are "
                             "valence and conduction bands. Use this when the top of the valence band is "
                             "not at 0. Default value is 0.")
    parser.add_argument("--nk_band", type=int, default=1000,
                        help="Total number of k points for the bandstructure plot. If adaptive_grid_length is set to 0 "
                             "this will be the number of points between each symmetry point")
    parser.add_argument("--nv", type=int, default=np.inf,
                        help="""The number of valence bands included in the calculations. By default we take all valence 
                             bands""")
    parser.add_argument("--nc", type=int, default=np.inf,
                        help="""The number of conduction bands included in the calculations. By default we take all 
                             conduction bands""")
    parser.add_argument("--data_file_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the data file")
    parser.add_argument("--unit_cell_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the unit file", )
    parser.add_argument("--save", type=int, default=1,
                        help="Indicate whether or not the save the results. True on default.")

    args = parser.parse_args()
    return args


def gaussian_broadening(energy, full_width_half_maximum):
    gaussian = 2 / full_width_half_maximum * np.sqrt(np.log(2) / np.pi) * np.exp(-4 * np.log(2) *
                                                                                 (energy/full_width_half_maximum)**2)
    return gaussian


def Calculate_DOS(energy, eigvals, fwhm):
    length = len(energy)
    dos = np.zeros(length)
    for index, E in enumerate(energy):
        eigvals_close = eigvals[abs(eigvals - E) < 2 * fwhm]
        dos[index] = np.sum(gaussian_broadening(eigvals_close - E, fwhm))
    return dos


def main():
    r = CoordSys3D("r")
    args = parse_arguments()
    unit_cell = UnitCell.from_file("./unit_cells/{}.json".format(args.material + args.unit_cell_suffix), r)
    k_values = unit_cell.k_values_ibz(args.nk)
    data_file = args.material + args.data_file_suffix
    if args.wannier:  # get the eigenvalues from existing wannier90 Hamiltonians
        data = Fitting_data('matrices/' + data_file)
        input_matrices_R = data.matrices
        model = TightBinding(unit_cell, r, args.order)
        _, _, eigenvalues, _ = model.matrices_R_to_k(input_matrices_R, k_values, r)
    else:
        model = TightBinding(unit_cell, r, args.order)  # Uses Tightbinding class from /tight_binding/model.py
        matrix = model.construct_hamiltonian()
        assert matrix.is_hermitian
        pprint(matrix)
        parameters_collection = ParameterCollection.from_file(  # Gets the energies and overlap integrals.
            "tight_binding_parameters/{}.json".format(data_file), str(args.order))
        eigenvalues = model.get_energy_eigenvalues({**parameters_collection.energies,
                                                    **parameters_collection.overlap_integrals,
                                                    **parameters_collection.soc_coefficients}, k_values)

    eigenvalues = eigenvalues[model.find_chosen_bands(eigenvalues, args.nc, args.nv, args.Ef, printing=True), :]
    eigenvalues = model.shift_eigenvalues(eigenvalues, args.Ef)

    """""
    Calculate the density of states
    """""

    Emax = np.max(eigenvalues)
    Emin = np.min(eigenvalues)
    E = np.linspace(Emin, Emax, args.N)
    dE = (Emax - Emin) / (args.N - 1)

    print(time.perf_counter())
    fwhm = args.fwhm
    DOS = Calculate_DOS(E, eigenvalues, fwhm)
    DOS = DOS / len(k_values.T)  # normalize to make it independent on our grid size
    print(time.perf_counter())

    print('Number of states calculated from the DOS: ' + str(
        np.sum(DOS * dE)))  # this should be roughly equal to the number of bands

    if args.symmetry_points:
        bandstructures, labels, n_pos_symmetry_points = gen_bandstructure.main(r, args)
        parameter_keys = ['nk', 'N', 'fwhm', 'n_pos_symmetry_points', 'labels']
        parameter_values = [args.nk, args.N, fwhm, n_pos_symmetry_points, labels]
        data_keys = ['E', 'DOS', 'Bandstructures']
        data_values = [E, DOS, bandstructures]
        data = data_processing.to_dict(args.material, data_keys, data_values, parameter_keys, parameter_values)
        data_processing.generate_plots(data_list=[data], number_of_plots=1)
    else:
        parameter_keys = ['nk', 'N', 'fwhm']
        parameter_values = [args.nk, args.N, fwhm]
        data_keys = ['E', 'DOS']
        data_values = [E, DOS]
        data = data_processing.to_dict(args.material, data_keys, data_values, parameter_keys, parameter_values)
        data_processing.generate_plots(data_list=[data], number_of_plots=1)
    plt.show()

    return


if __name__ == "__main__":
    main()
