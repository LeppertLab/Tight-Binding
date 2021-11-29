#!/usr/bin/env python
"""Fit a Tight Binding model to existing bandstructures."""
from tight_binding.model import TightBinding
from sympy.vector import CoordSys3D
from tight_binding.objects import UnitCell
import fitting.fit as fit
from fitting import Fitting_data
from multiprocessing import Pool
from functools import partial
from itertools import chain
from tqdm import tqdm
import pandas as pd
from os import mkdir, path
import json
from datetime import datetime
import argparse
from utils.JSONEncoder import CustomEncoder
import time

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

SIMULATION_BASE_DIRECTORY = "simulations/"


def create_simulation_directory() -> str:
    """Create directory for simulation information and results."""
    now = datetime.now()
    dirname = path.join(SIMULATION_BASE_DIRECTORY, now.strftime("%Y%m%d-%H%M%S"))
    mkdir(dirname)
    return dirname


DEFAULT_CPU_THREADS = 8
DEFAULT_BASIN_ITERATIONS = 50
DUMPFILE_NAME = "simulations/fit_lowest_f " + time.strftime(
    "%Y-%m-%d %H-%M-%S") + '.json'  # this file contains the best parameters of the last time you ran this code.


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit a Tight Binding model to an existing bandstructure.")
    parser.add_argument("material",
                        help="Material for which you want to fit a bandstructure.")
    parser.add_argument("order", type=int,
                        help="Maximum order of nearest neighbours that are included.")
    parser.add_argument("--basin", type=int, default=DEFAULT_BASIN_ITERATIONS,
                        help=f"How many basinhopping interations you'd like to perform. "
                             f"Default: {DEFAULT_BASIN_ITERATIONS}")
    parser.add_argument("--threads", type=int, default=DEFAULT_CPU_THREADS,
                        help=f"The amount of threads to use for the fit. Default: {DEFAULT_CPU_THREADS}")
    parser.add_argument("--data_file_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the data file")
    parser.add_argument("--unit_cell_suffix", type=str, default="",
                        help="A suffix to attach to the filename for the unit file")
    parser.add_argument("--high_symmetry_only", type=int, default=False,
                        help="Determines if the fit is only done at the high symmetry points to speed up the "
                             "calculations")
    parser.add_argument("--nk", type=int,
                        help="Number of k points.")
    parser.add_argument("--symmetry_points", metavar="symmetry-point", nargs="*", type=str,
                        help="Path of symmetry points through which we generate bandstructures")
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
    parser.add_argument("-T", "--temperature", type=float, default=0.25,
                        help="""The temperature for the accept/reject criterion of the basinhopping algorithm. Higher
                        temperatures means that a step is more likely to be accepted if the trial_f is higher.
                        Lower temperature makes a step less likely to be accepted if the trial_f is higher.
                        A temperature of 0 means that the step will only be accepted if the trial_f is lower than
                        the starting value of f (start_f). Probability is calculated as: 
                        exp(-(func(f_new)-func(f_old))/T)""")
    parser.add_argument("-s", "--stepsize", type=float, default=1,
                        help="""Starting stepsize for the basinhopping algorithm. The value of the stepsize is the
                        maximum value with which a parameter can change in value within 1 step. The stepsize will be
                        changed automatically to get to a target accept rate of 0.5.""")
    parser.add_argument("--wannier", type=int, default=0,
                        help="Specify whether you want to use matrices from wannier90")
    return parser.parse_args()


def store_simulation_parameters(simulation_directory, **kwargs):
    """Store simulation parameters in JSON format."""
    with open(path.join(simulation_directory, "info.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(kwargs, cls=CustomEncoder))


def generate_configurations(orbitals, threads, path_guesses, order):
    """This function gives a permutation over all possible combinations with the given input parameters (optional).
    This function randomly assigns a value to the non specified parameters."""
    if path.exists(path_guesses):  # Get (some of the) parameter guesses from the file if it exists.
        f = open(path_guesses, encoding='utf-8')
        data = json.load(f)
        f.close()
        guesses = data["order"][str(order)]
    else:  # Randomly generate all of the parameter guesses
        guesses = {}
    rng = np.random.default_rng()
    configurations = []
    for index in range(threads // 2):
        configuration = {}
        for symbol in orbitals:
            if str(symbol) in guesses.keys():
                configuration[symbol] = guesses[str(symbol)]
            else:
                configuration[symbol] = rng.uniform(-5, 5)
        configurations.append(configuration)
    return configurations


def main():
    simulation_parameters = parse_arguments()

    unit_cell = simulation_parameters.material + simulation_parameters.unit_cell_suffix
    data_file = simulation_parameters.material + simulation_parameters.data_file_suffix
    if simulation_parameters.wannier == 1:
        data_file = 'matrices/' + data_file
        print('Fitting to band structures generated from Wannier Hamiltonians.')
    else:
        data_file = 'bandstructures/' + data_file
        print('Fitting to band structures.')

    data = Fitting_data(data_file, Ef=simulation_parameters.Ef)

    r = CoordSys3D("r")

    unit_cell_path = "./unit_cells/{}.json".format(unit_cell)
    with open(unit_cell_path, encoding="utf-8") as f:
        unit_cell_dict = json.loads(f.read())
    unit_cell = UnitCell.from_dict(unit_cell_dict, r)

    model = TightBinding(unit_cell, r, simulation_parameters.order)
    matrix = model.construct_hamiltonian()
    pprint(matrix)

    symmetry_points = data.get_symmetry_points()
    nk = simulation_parameters.nk

    if data.type == 'bandstructures':
        heuristic = [  # Functions with their assigned weight
            (fit.Basinhopping_funcs.heuristic_least_squares, 1),
            (fit.Basinhopping_funcs.boundary_check, 100),
        ]
        eigenvalues = data.load_bands(symmetry_points)
        points_per_direction = data.get_points_per_direction()
        n_pos_symmetry_points = list(np.insert(np.cumsum(points_per_direction), 0, 0))
        k_values, n_pos_symmetry_points = unit_cell.get_k_values(number_of_k_values=np.sum(points_per_direction),
                                                                 chosen_symmetry_points=symmetry_points,
                                                                 n=n_pos_symmetry_points,
                                                                 adaptive=1 if len(
                                                                     set(points_per_direction)) > 1 else 0)
        assert matrix.shape[0] == eigenvalues.shape[0], str(matrix.shape[0]) + ' != ' + str(eigenvalues.shape[0])
    elif data.type == 'wannier':
        heuristic = [  # Functions with their assigned weight
            (fit.Basinhopping_funcs.heuristic_least_squares, 1),
            (fit.Basinhopping_funcs.boundary_check, 100),
        ]
        if simulation_parameters.symmetry_points:
            k_values, n_pos_symmetry_points = unit_cell.get_k_values(nk, symmetry_points, adaptive=1)
        else:
            k_values, n_pos_symmetry_points = unit_cell.k_values_ibz(nk=nk), None
        input_matrices_R = data.matrices  # real space Hamiltonians from Wannier90
        matrices_k, nabla_matrices_k, eigenvalues, eigenvectors = model.matrices_R_to_k(input_matrices_R, k_values, r)
        assert matrix.shape == matrices_k[:, :, 0].shape, str(matrix.shape) + ' != ' + str(matrices_k[:, :, 0].shape)
    else:
        raise NotImplementedError('Could not determine what to fit to.')

    iscdband = model.find_conduction_band(eigenvalues - simulation_parameters.Ef)  # find conduction bands
    eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]  # get valence bands
    shift = np.max(eigenvalues_v)  # determine valence band maximum
    eigenvalues = eigenvalues - shift  # shift valence band maximum to zero

    chosen_bands = model.find_chosen_bands(eigenvalues, simulation_parameters.nc, simulation_parameters.nv,
                                           Ef=0, printing=True)  # find the bands which we fit to
    eigenvalues = eigenvalues[chosen_bands, :]  # get the eigenvalues of the chosen bands

    with open(DUMPFILE_NAME, 'w+') as file:
        json.dump({"material": simulation_parameters.material, "order": simulation_parameters.order,
                   "nc": simulation_parameters.nc, "nv": simulation_parameters.nv,
                   "temperature": simulation_parameters.temperature, "stepsize": simulation_parameters.stepsize,
                   "f": np.inf, "parameters": {}},
                  file)

    energy_symbols = model.energy_symbols
    overlap_symbols = model.energy_integral_symbols
    soc_symbols = model.soc_symbols

    params = list(chain(energy_symbols, overlap_symbols, soc_symbols))
    params_dict = {'energy_symbols': energy_symbols, 'overlap_symbols': overlap_symbols, 'soc_symbols': soc_symbols}

    partialfunc = partial(
        fit.fit_bandstructure,  # function to execute with below its input parameters
        heuristic, model, matrix, params, k_values, eigenvalues, chosen_bands, simulation_parameters, DUMPFILE_NAME,
        params_dict)

    print("SOC", unit_cell.spin_orbit_coupling)
    print("Symbols", matrix.free_symbols)

    path_guesses = 'parameters_guesses/' + simulation_parameters.material + '.json'
    configurations = generate_configurations(params, simulation_parameters.threads, path_guesses,
                                             simulation_parameters.order)

    print("We'll be guessing the following starting parameters:")
    for configuration in configurations:
        print(configuration)

    pool = Pool(simulation_parameters.threads // 2)  # Scipy basinhopping uses two threads per pool
    list_of_results = tqdm(
        pool.imap(partialfunc, configurations), total=len(configurations)
    )
    pool.close()

    dataset = []
    info_data = {"lowest_f": [], "temperature": [], "stepsize": [], "adaptive_step_interval": [],
                 "target_accept_rate": [], "accept_rate": [], "iteration_lowest_f": [], "minimization_failures": [],
                 "not_moved_count": [], "parameters": [], "parameter_values": [], "elapsed_time": []}
    param_values = []
    for i, configuration in enumerate(list_of_results):
        fit_data = configuration[0]
        param_values.append(configuration[1])
        data = dict(zip(fit_data["parameters"], fit_data["parameter_values"]))
        data["lowest_f"] = fit_data["lowest_f"]
        dataset.append(data)
        info_data["lowest_f"].append(fit_data["lowest_f"])
        info_data["temperature"].append(fit_data["temperature"])
        info_data["stepsize"].append(fit_data["stepsize"])
        info_data["adaptive_step_interval"].append(fit_data["adaptive_step_interval"])
        info_data["iteration_lowest_f"].append(fit_data["iteration_lowest_f"])
        info_data["accept_rate"].append(fit_data["accept_rate"])
        info_data["target_accept_rate"].append(fit_data["target_accept_rate"])
        info_data["minimization_failures"].append(fit_data["minimization_failures"])
        info_data["not_moved_count"].append(fit_data["not_moved_count"])
        info_data["parameters"].append(fit_data["parameters"])
        info_data["parameter_values"].append(fit_data["parameter_values"])
        info_data["elapsed_time"].append(fit_data["elapsed_time"])

    simulation_directory = create_simulation_directory()
    df = pd.DataFrame(dataset)
    df.to_csv("{basedir}/data.csv".format(basedir=simulation_directory), index=False)

    store_simulation_parameters(simulation_directory,
                                arguments=simulation_parameters.__dict__,
                                version=0.4,
                                k_path={"symmetry_points": symmetry_points,
                                        "n_pos_symmetry_points": n_pos_symmetry_points},
                                unit_cell=unit_cell_dict,
                                data_file=data_file,
                                heuristic=heuristic,
                                fit_info=info_data,
                                )

    value, index = min((value, index) for (index, value) in enumerate(info_data["lowest_f"]))
    fitted_eigenvalues = model.get_energy_eigenvalues(
        dict(zip([str(x) for x in info_data['parameters'][index]], param_values[index])), k_values)
    fitted_eigenvalues = fitted_eigenvalues[chosen_bands]
    for band in fitted_eigenvalues:
        plt.plot(band)
    for band in eigenvalues:
        plt.plot(band, marker='o', linestyle='', color='black', markersize=1)
    error = fitted_eigenvalues - eigenvalues
    print('f = ' + str(np.average(np.abs(np.power(error, 2)))))
    plt.show()
    return


if __name__ == "__main__":
    main()
