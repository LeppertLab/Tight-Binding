#!/usr/bin/env python
"""Fit a Tight Binding model to existing bandstructures."""
from tight_binding.model import TightBinding
from tight_binding.objects import UnitCell
from sympy.vector import CoordSys3D
from pprint import pprint

from tight_binding_parameters import ParameterCollection
import argparse
import numpy as np
import matplotlib.pyplot as plt
import data_processing


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Generate a bandstructure using a Tight Binding model for a certain material""")
    parser.add_argument("material",
                        help="Material for which to generate a bandstructure")
    parser.add_argument("order", type=int,
                            help="Maximum order of nearest neighbours that are included. Only used when wannier = 0")
    parser.add_argument("symmetry_points_minimal", metavar="symmetry-point", nargs=2, type=str,
                        help="Symmetry points through which to generate bandstructures")
    parser.add_argument("symmetry_points_additional", metavar="symmetry-points", nargs="*", type=str,
                        help="Additional symmetry points")
    parser.add_argument("--adaptive_grid_length", type=int, default=1,
                        help="Adaptive grid length determines the number of points between symmetry points based on the"
                             " distance between the respective symmetry points.")
    parser.add_argument("--nk", type=int, default=1000,
                        help="Total number of k points. If adaptive_grid_length is set to 0 this will be the number of "
                             "points between each symmetry point")
    parser.add_argument("--wannier", type=int, default=0,
                        help="Specify whether you want to use matrices from wannier90")
    parser.add_argument("--Ef", type=float, default=0,
                        help="""Roughly specify the location of the Fermi level. Used to determine which bands are
                             valence and conduction bands. Use this when the top of the valence band is 
                             not at 0. Default value is 0.""")
    parser.add_argument("--Ef2", type=float, default=0,
                        help="""Fermi level for the external bandstructures.""")
    parser.add_argument("--nv", type=int, default=np.inf,
                        help="""The number of valence bands included in the calculations. By default we take all valence 
                             bands""")
    parser.add_argument("--nc", type=int, default=np.inf,
                        help="""The number of conduction bands included in the calculations. By default we take all 
                             conduction bands""")
    parser.add_argument("--unit_cell_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the unit file")
    parser.add_argument("--data_file_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the data file")
    parser.add_argument("--bandstructure_file_suffix", type=str, default="",
                        help="Suffix for the JSON file containing the band structures from wannier90. If this is given"
                             "the calculated band structures will be compared to these external band structures")
    parser.add_argument("--labels", type=str, nargs=2, default=["fitted", "input"],
                        help="Labels for the fitted and external band structure (from wannier90)")
    parser.add_argument("--save", type=int, default=1,
                        help="Indicate whether or not the save the results. True on default.")
    args = parser.parse_args()
    args.symmetry_points = (args.symmetry_points_minimal + args.symmetry_points_additional)
    return args


def bandstructure_JSON(path):
    """Get the band structure from an external file. We can compare our calculated band structure to this one."""
    unit_cell = UnitCell.from_file("./unit_cells/{}.json".format(args.material + args.unit_cell_suffix), r)
    model = TightBinding(unit_cell, r, args.order)
    eigenvalues, symmetry_points, n_pos_symmetry_points = data_processing.bandstructures_JSON(path)
    eigenvalues = np.array(eigenvalues)
    eigenvalues = eigenvalues[model.find_chosen_bands(eigenvalues, args.nc, args.nv, args.Ef2, printing=True), :]
    eigenvalues = model.shift_eigenvalues(eigenvalues, args.Ef2)
    return eigenvalues


def main(r, args):
    if __name__ != "__main__":  # Used when calling from density_of_states.py since nk is defined differently there
        args.nk = args.nk_band

    unit_cell = UnitCell.from_file("./unit_cells/{}.json".format(args.material + args.unit_cell_suffix), r)
    model = TightBinding(unit_cell, r, args.order)
    data_file = args.material + args.data_file_suffix
    k_values, n_pos_symmetry_points = unit_cell.get_k_values(args.nk, args.symmetry_points,
                                                             adaptive=args.adaptive_grid_length)
    if args.wannier:
        from fitting import Fitting_data
        data = Fitting_data('matrices/' + data_file)
        matrices = data.matrices
        matrices_k, nabla_input_matrices_k, eigenvalues, input_eigenvectors = model.matrices_R_to_k(matrices, k_values,
                                                                                                    r)
    else:
        matrix = model.construct_hamiltonian()
        pprint(matrix)
        parameters_collection = ParameterCollection.from_file(
            "tight_binding_parameters/{}.json".format(data_file), str(args.order))
        eigenvalues = model.get_energy_eigenvalues({**parameters_collection.energies,
                                                    **parameters_collection.overlap_integrals,
                                                    **parameters_collection.soc_coefficients},
                                                   k_values)
    eigenvalues = eigenvalues[model.find_chosen_bands(eigenvalues, args.nc, args.nv, args.Ef, printing=True), :]
    # eigenvalues = model.shift_eigenvalues(eigenvalues, args.Ef)
    return eigenvalues, args.symmetry_points, n_pos_symmetry_points


if __name__ == "__main__":
    r = CoordSys3D("r")
    args = parse_arguments()
    eigenvalues, _, n_pos_symmetry_points = main(r, args)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Band structure: {material}".format(material=args.material))

    if args.save == 1:
        parameter_keys = ['nk', "n_pos_symmetry_points", "labels"]  # parameters that we want to save
        parameter_values = [args.nk, n_pos_symmetry_points, args.symmetry_points]  # values for the parameter
        data_keys = ['Bandstructures']  # data which we want to save
        data_values = [eigenvalues]  # values for the data
        data = data_processing.to_dict(args.material, data_keys, data_values, parameter_keys, parameter_values)
        data_processing.save_as_json(data)

    x = np.linspace(0, args.nk, args.nk)
    if args.bandstructure_file_suffix:
        eigenvalues_external = bandstructure_JSON(
            path='D:/Documents/School/Master/Graduation_assignment/Workspace/fitting/fit_data/bandstructures/' +
                 args.material + args.bandstructure_file_suffix + '.json')
        for index, band in enumerate(eigenvalues):
            if index == 0:
                plt.plot(x, band, label=args.labels[0])
            else:
                plt.plot(x, band)
        for index, band in enumerate(eigenvalues_external):
            x2 = np.linspace(0, np.shape(eigenvalues_external)[1], np.shape(eigenvalues_external)[1])
            if index == 0:
                plt.plot(x2, band, marker='o', linestyle='', color='black', label=args.labels[1])
            else:
                plt.plot(x2, band, marker='o', linestyle='', color='black')
        plt.legend(loc='upper right')
        plt.xticks(n_pos_symmetry_points, [f"${symmetry_point}$" for symmetry_point in args.symmetry_points])
        plt.ylabel('Energy (eV)')
        plt.title('Band structure: ' + args.material)
        plt.grid()
        fig2 = plt.figure()
        from visualisation.plot import visualize_fit
        visualize_fit(fig2, n_pos_symmetry_points, eigenvalues_external, eigenvalues, args.symmetry_points)
    else:
        from visualisation.plot import plot_band_structure
        plot_band_structure(ax, n_pos_symmetry_points, eigenvalues, args.symmetry_points)
    plt.show()
