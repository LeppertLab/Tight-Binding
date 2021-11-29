from pathlib import Path
from tkinter.filedialog import askopenfilename
import numpy as np
import json
import codecs
import argparse


def parse_arguments() -> argparse.Namespace:  # Handles the input parameters from the command line.
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Saves the fit to /tight_binding_parameters"""
    )
    parser.add_argument(
        "--suffix",
        help="Suffix for the parameters/unitcell file to avoid overwriting previous parameters",
        type=str,
        default="_fit"
    )
    arguments = parser.parse_args()
    return arguments


def save_fit_parameters(args):

    path_fit = Path(askopenfilename(initialdir='simulations'))
    fitted_parameters = np.genfromtxt(open(path_fit, encoding="utf-8"), delimiter=',', dtype=str)
    info = json.load(open(str(path_fit.parent / 'info.json')))
    path_parameters = Path('tight_binding_parameters/' + info['arguments']['material'] +
                           info['arguments']['unit_cell_suffix'] + '_' + args.suffix + '.json')
    path_unitcell = Path('unit_cells/' + info['arguments']['material'] + info['arguments']['unit_cell_suffix'] + '.json')
    assert Path.exists(path_unitcell)

    from sympy.vector import CoordSys3D
    from tight_binding.objects import UnitCell
    from tight_binding.model import TightBinding
    r = CoordSys3D("r")
    unit_cell = UnitCell.from_file(path_unitcell, r)
    model = TightBinding(unit_cell, r, info['arguments']['order'])
    model.construct_hamiltonian()

    order = str(info['arguments']['order'])
    if path_parameters.exists():
        parameters_json = json.load(open(path_parameters, encoding="utf-8"))
    else:
        parameters_json = {'order': {}}
        parameters_json['order'][order] = {}
    parameters_json['order'][order] = {}
    parameters_json['order'][order]['energies'] = {}
    parameters_json['order'][order]['overlap_integrals'] = {}
    if unit_cell.spin_orbit_coupling:
        parameters_json['order'][order]['soc_coefficients'] = {}

    list_lowest_f = [float(x) for x in fitted_parameters[:, -1][1:]]
    lowest_f, index_lowest_f = min((val, idx+1) for (idx, val) in enumerate(list_lowest_f))
    fitted_parameters = fitted_parameters[[0, index_lowest_f], :]
    sorted_indices = np.zeros(len(fitted_parameters.T))
    for index, parameter in enumerate(fitted_parameters[0]):
        try:
            int(parameter[-1])
            sorted_indices[index] = int(int(parameter[-1]))
        except ValueError:
            if parameter[0:6] == "lambda":
                sorted_indices[index] = np.inf
            else:
                sorted_indices[index] = 0
    sorted_order = sorted(range(len(sorted_indices)), key=lambda x: sorted_indices[x])
    fitted_parameters[0] = fitted_parameters[0, sorted_order]
    fitted_parameters[1] = fitted_parameters[1, sorted_order]
    for [parameter, value] in fitted_parameters.T:
        if parameter in [str(x) for x in model.energy_symbols]:  # eigenvalue unperturbed Hamiltonian
            parameters_json['order'][order]['energies'][parameter[2:]] = float(value)
        elif parameter in [str(x) for x in model.energy_integral_symbols]:  # overlap parameter
            parameters_json['order'][order]['overlap_integrals'][parameter] = float(value)
        elif parameter in [str(x) for x in model.soc_symbols]:  # overlap parameter
            parameters_json['order'][order]['soc_coefficients'][parameter] = float(value)
    if args.suffix is not None:
        path_parameters = path_parameters.parent / (info['arguments']['material'] +
                                                    info['arguments']['unit_cell_suffix'] + args.suffix + '.json')
    dumpfile = codecs.open(path_parameters, 'w', encoding='utf8')
    json.dump(parameters_json, dumpfile, indent=4, ensure_ascii=False)
    dumpfile.close()
    return


if __name__ == '__main__':
    args = parse_arguments()
    save_fit_parameters(args)
