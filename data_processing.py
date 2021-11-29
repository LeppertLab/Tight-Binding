import json
from tkinter.filedialog import askopenfilename
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib as mpl
import time
import os

mpl.rcParams['lines.markersize'] = 1


def to_dict(material, data_keys, data_values, parameter_keys, parameter_values):
    """Stores the plot information in a dictionary that can be saved to a JSON file"""
    data = {'material': material, 'parameters': {}}
    for index, parameter_key in enumerate(list(parameter_keys)):
        data['parameters'][parameter_key] = parameter_values[index]
    for index, data_key in enumerate(list(data_keys)):
        if isinstance(data_values[index], np.ndarray):
            data[data_key] = np.ndarray.tolist(data_values[index])
        else:
            data[data_key] = data_values[index]
    return data


def save_as_json(data, location='results'):
    """Saves the data in a JSON file for future access (for example to compare to other data)"""
    base = Path(location)
    name = time.strftime("%Y-%m-%d %H-%M-%S") + '.json'
    path = base / name
    with open(path, 'w') as f:
        json.dump(data, f)
    return


def load_csv():
    """Loads data from a csv file"""
    path = askopenfilename()
    with open(path) as f:
        raw_data = csv.reader(f, delimiter=',')
    data = {}
    x = []
    y = []
    for index, row in enumerate(raw_data):
        if index > 0:
            x.append(float(row[0]))
            y.append(float(row[1]))
    filename = Path(path).stem
    data['x'] = x
    data['y'] = y
    data['source'] = filename
    return data


def Optical_properties_to_JSON(Type="RPA"):
    """Function that I use to convert the optical properties from the RPA spectrum to a usable JSON file"""
    path = askopenfilename()
    with open(path) as f:
        raw_data = np.genfromtxt(f)
    data = {"path": path, "type": Type, "Ep_range": list(raw_data[:, 0]), '\u03B5_i': list(raw_data[:, 1]),
            '\u03B5_r': list(raw_data[:, 2]), "JDOS": list(raw_data[:, 3])}
    save_as_json(data, location='external_results')
    return


def Wannier_dat_to_JSON():
    """Function that I use to convert the output Hamiltonians from Wannier90 to a usable JSON file"""
    import math
    path = askopenfilename()  # select the material_hr file
    with open(path) as f:
        row_header_end = np.inf
        degeneracy = []
        for index, line in enumerate(f):
            if index == 1:
                N = int(line)  # Dimension of the matrix
            elif index == 2:
                nrpts = int(line)  # Number of Wigner-Seitz grid-points
                row_header_end = index + math.ceil(
                    nrpts / 15) + 1  # Row at which the header ends. Skip to avoid problems.
            elif index == row_header_end:
                break
            elif index >= 3:
                degeneracy = degeneracy + list(line[4::5])
        degeneracy = list(map(int, degeneracy))
        raw_data = np.genfromtxt(f, skip_header=row_header_end)
    data = [{"path": path, "type": "wannier"}, {"matrices": {}}]
    matrix_real = np.empty([N, N])
    matrix_imag = np.empty([N, N])
    index = 0
    for row in raw_data:
        R = row[0:3]
        m = int(row[3]) - 1
        n = int(row[4]) - 1
        matrix_real[m, n] = row[5] / degeneracy[index]
        matrix_imag[m, n] = row[6] / degeneracy[index]
        if m == n == N - 1:
            data[1]["matrices"][str(tuple(int(x) for x in R))] = [matrix_real.tolist(), matrix_imag.tolist()]
            matrix_real = np.empty([N, N])
            matrix_imag = np.empty([N, N])
            index += 1
    save_as_json(data, location='fitting/fit_data/matrices')
    return


def external_bands_to_JSON(SYMMETRY_LINES, N_LINE, N_VALENCE, N_CONDUCTANCE, EF=0):
    """This is a function I use to convert data from external band structures to a format that can be used by
    fit_bandstructure.py. The JSON file should be moved to the 'bandstructures' folder (from results)"""

    def get_bands(data, raw_data, start_index, stop_index, n_per_band, n_line, symmetry_lines, n_conductance, Ef,
                  all_bands=True):
        n_band, conductance_count, valence_count, number_of_bands, reached_end = 0, 0, 0, 0, 0
        for index, symmetry_line in enumerate(symmetry_lines):
            data.append({"direction": symmetry_lines[index], "datapoints": []})
            n_start = sum([x for x in n_line[:index]])  # start index of the symmetry line
            n_end = sum([x for x in n_line[:index + 1]])  # end index of the symmetry line
            n_band, conductance_count, valence_count, number_of_bands, reached_end = 0, 0, 0, 0, 0
            while reached_end == 0:
                if start_index < n_band <= stop_index or all_bands:
                    datapoints = list(raw_data[n_band + n_start:n_band + n_end, 1])
                    data[-1]['datapoints'].append(datapoints)
                    if datapoints[0] >= Ef:
                        conductance_count += 1  # count the number of conduction bands
                    else:
                        valence_count += 1  # count the number of valence bands
                    if conductance_count == n_conductance:
                        stop_index = n_band  # the index at which we need to stop (for the second time we run this)
                    number_of_bands += 1  # count the total number of bands that we include
                n_band += n_per_band  # go to the next band
                if n_band >= len(raw_data):
                    reached_end = 1
        data[0]['number_of_bands'] = number_of_bands
        data[0]['valence_count'] = valence_count
        data[0]['conductance_count'] = conductance_count
        if stop_index == np.inf:
            stop_index = n_band - n_per_band
        return data, stop_index

    N_PER_BAND = sum(N_LINE)  # total number of points per band
    PATH = askopenfilename()
    with open(PATH) as FILE:
        RAW_DATA = np.genfromtxt(FILE)
    ALL_DATA = [{"path": PATH, "type": "bandstructures"}]  # contains all bands
    START_INDEX, STOP_INDEX = 0, np.inf
    ALL_DATA, STOP_INDEX = get_bands(ALL_DATA, RAW_DATA, START_INDEX, STOP_INDEX, N_PER_BAND, N_LINE, SYMMETRY_LINES,
                                     N_CONDUCTANCE, EF, all_bands=True)
    N_VALENCE = ALL_DATA[0]['valence_count'] if N_VALENCE > ALL_DATA[0]['valence_count'] else N_VALENCE
    N_CONDUCTANCE = ALL_DATA[0]['conductance_count'] if \
        N_CONDUCTANCE > ALL_DATA[0]['conductance_count'] else N_CONDUCTANCE
    N_BANDSTRUCTURE = np.sum(N_LINE)
    START_INDEX = (STOP_INDEX / N_BANDSTRUCTURE - (N_VALENCE + N_CONDUCTANCE)) * N_BANDSTRUCTURE

    DATA = [{"path": PATH, "type": "bandstructures"}]  # only contains the relevant band structures
    DATA, _ = get_bands(DATA, RAW_DATA, START_INDEX, STOP_INDEX, N_PER_BAND, N_LINE, SYMMETRY_LINES,
                        N_CONDUCTANCE, EF, all_bands=False)
    save_as_json(DATA, location='fitting/fit_data/bandstructures')
    return


def bandstructures_JSON(path):
    """This function returns the bandstructures of the JSON files used by fit_bandstructure.py (for example the
    bandstructures obtained from DFT calculations)"""
    with open(path) as f:
        raw_data = json.load(f)
    number_of_bands = len(raw_data[1]['datapoints'])
    bandstructures = [[] for x in range(number_of_bands)]
    directions = []
    number_of_points = []
    for data in raw_data:
        if 'datapoints' in [key for key in data.keys()]:
            directions.append(data["direction"])
            number_of_points.append(len(data["datapoints"][0]))
            for index, row in enumerate(data["datapoints"]):
                if index < number_of_bands:
                    bandstructures[index] += row
                else:
                    break

    n_pos_symmetry_points = np.cumsum(number_of_points)
    n_pos_symmetry_points = [0] + [int(x) for x in n_pos_symmetry_points]

    symmetry_points = [directions[0][0]]
    prev_direction = [directions[0][0], directions[0][0]]
    for direction in directions:
        if direction[0] != prev_direction[1]:
            symmetry_points[-1] = symmetry_points[-1] + ',' + direction[0]
        symmetry_points.append(direction[1])
        prev_direction = direction
    return bandstructures, symmetry_points, n_pos_symmetry_points


def load_external_json():
    """Loads JSON files generated by WebPlotDigitizer to convert it into usable dictionaries"""
    filename = askopenfilename(initialdir='D:\Documents\School\Master\Graduation_assignment\Plot_data\data')
    with open(filename) as f:
        raw_data_set = json.load(open(f))
    data_sets = []
    for raw_data in raw_data_set["datasetColl"]:
        data_set = {"name": raw_data["name"]}  # dataset which you want to create
        data = raw_data["data"]
        x = []
        y = []
        for element in data:
            x.append(element["value"][0])
            y.append(element["value"][1])
        sorted_indices = np.argsort(x)  # search for the indices in order of increasing x
        x = [x[i] for i in sorted_indices]
        y = [y[i] for i in sorted_indices]
        data_set["external"] = True  # Indicates that the structure is not from this python code
        data_set["x"] = x
        data_set["y"] = y
        data_sets.append(data_set)
    return data_sets


def prepare_plot(material, measured_variable):
    """Gives a title, xlabel, and ylabel to the plots depending on the measured variable"""
    plt.grid(True)
    if measured_variable == '\u03B5_i':
        plt.title('Imaginary dielectric function: ' + material)
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('Im[\u03B5(E)]')
    elif measured_variable == '\u03B5_r':
        plt.title('Real dielectric function: ' + material)
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('Re[\u03B5(E)]')
    elif measured_variable == '\u03B1':
        plt.title('Absorption spectrum: ' + material)
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('\u03B1(cm$^{-1}$)')
        plt.grid(True, which='minor', alpha=0.2)
        plt.minorticks_on()
    elif measured_variable == 'JDOS':
        plt.title('JDOS: ' + material)
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('JDOS (a.u.)')
    elif measured_variable == 'DOS':
        plt.title('Density of states: ' + material)
        plt.ylabel('Normalized DOS')
        plt.xlabel('Energy')
    elif measured_variable == "Bandstructures":
        plt.title('Band structure: ' + material)
        plt.ylabel('Energy (eV)')
        plt.grid(True, which='minor', alpha=0.2)
        plt.minorticks_on()
    else:
        plt.title(str(measured_variable) + ': ' + material)
        plt.ylabel(str(measured_variable))
        plt.grid(True, which='minor', alpha=0.2)
        plt.minorticks_on()
    return


def Compare_to_Webplotdigitizer(label_external="source"):
    """Used to compare results from the model to results obtained with Webplotdigitizer (from papers for example)"""
    with open(askopenfilename(initialdir='results')) as f:
        data = json.load(f)
    data_external_list = load_external_json()

    if 'DOS' in data.keys():
        if 'Bandstructures' in data.keys():
            fig, (axDOS, axBand) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
            for band in data['Bandstructures']:
                axBand.plot(band)
        else:
            fig, axDOS = plt.subplots(1, 1)
        axDOS.plot(data['DOS'], data['E'], label="model")
    elif 'Bandstructures' in data.keys():
        fig, axBand = plt.subplots(1, 1)
        for index, band in enumerate(data['Bandstructures']):
            if index == 0:
                axBand.plot(band, label='model')
            else:
                axBand.plot(band)

    for data_external in data_external_list:
        plt.gca().set_prop_cycle(None)  # resets color cycle
        if "Bandstructures" in data_external['name'] and 'Bandstructures' in data.keys():
            data_external["x"] = [x * len(band) for x in data_external["x"]]
            axBand.plot(data_external["x"], data_external["y"], marker='o', linestyle='', color='black',
                        label=label_external)
            axBand.label_outer()
            axBand.grid(True)
            # axBand.set_ylim(data['E'][0] - 0.25, data['E'][-1] + 0.25)
            n_pos_symmetry_points = (data['parameters']['n_pos_symmetry_points'])
            plt.xticks(n_pos_symmetry_points,
                       [f"${symmetry_point}$" for symmetry_point in data['parameters']['labels']])
            axBand.set_title('Band structure: ' + data['material'])
            if 'DOS' not in data.keys():
                axBand.legend()
            axBand.grid(True)
            axBand.set_ylabel('Energy (eV)')
        elif "DOS" in data_external['name']:
            if axBand:
                axDOS.plot(data_external["y"], data_external["x"], label=label_external, marker='o', linestyle='',
                           color='black')
                axDOS.set_xlim(np.max(data["DOS"] + data_external["y"]) + 0.01, -0.01)
                axDOS.set_ylim(data["E"][0] - 0.25, data["E"][-1] + 0.25)
                axDOS.set_xlabel('DOS (states/eV)')
                axDOS.set_ylabel('Energy (eV)')
            else:
                axDOS.plot(data_external["x"], data_external["y"], label=label_external, marker='o', linestyle='',
                           color='black')
                axDOS.set_xlim(np.max(data["DOS"] + data_external["y"]) + 0.01, -0.01)
                axDOS.set_ylim(data["E"][0] - 0.25, data["E"][-1] + 0.25)
                axDOS.set_xlabel('DOS (states/eV)')
                axDOS.set_ylabel('Energy (eV)')
            axDOS.set_title('Density of states')
            axDOS.grid(True)
            axDOS.legend()
        elif '\\u03B1' in data_external['name']:
            prepare_plot(data['material'], '\u03B1')
            plt.semilogy(data['Ep_range'], data['\u03B1'], label='model')
            # plt.semilogy(data_external["x"], data_external["y"], label=label_external, marker='o', linestyle='',
            #          color='black')
            plt.semilogy(data_external["x"], data_external["y"], label=label_external, color='red')
            plt.legend()
        elif 'tme' in data_external['name']:
            tme = np.array(data['tme'])
            tme_external = np.array(data_external['y'])
            # tme = tme / np.max(tme) / len(tme)
            tme = tme / (np.sum(tme) / len(tme))
            # tme_external = tme_external / np.max(tme_external)
            tme_external = tme_external / (np.sum(tme_external) / len(tme_external))
            plt.plot(tme)
            plt.plot(np.array(data_external['x']) * len(tme), tme_external, color='red')
            n_pos_symmetry_points = (data['parameters']['n_pos_symmetry_points'])
            plt.xticks(n_pos_symmetry_points,
                       [f"${symmetry_point}$" for symmetry_point in data['parameters']['labels']])
            plt.title('Transition matrix elements: ' + data['material'])
            plt.legend(['model', 'source'])
            plt.ylabel('P$^{2}$(a.u.)')

    plt.show()
    return


def generate_plots(data_list=None, number_of_plots=1, labels=None, color_list=None):
    """Used to generate plots from the results that we got (from gen_bandstructure.py, density_of_states.py and
    dielectric_function.py)."""
    if color_list is None:
        color_list = ['b', 'r', 'g', 'm', 'c', 'y']
    if data_list is None:
        data_list = []
    if labels is None:
        labels = [None] * number_of_plots

    def plot_DOS_plus_bands(data_list):
        """Use this if we want to plot the density of states next to the band structures"""
        fig, (axDOS, axBand) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
        if len(data_list) <= 2:  # 2 or less plots
            for index, data in enumerate(data_list):
                if index == 0:
                    for index2, band in enumerate(data['Bandstructures']):
                        if index2 == 0:
                            axBand.plot(band, label=labels[0])
                        else:
                            axBand.plot(band)
                    axDOS.plot(data['DOS'], data['E'], label=labels[0], color='blue')
                if index == 1:
                    for index2, band in enumerate(data['Bandstructures']):
                        if index2 == 0:
                            axBand.plot(band, marker='o', linestyle='', color='black', label=labels[1])
                        else:
                            axBand.plot(band, marker='o', linestyle='', color='black')
                    axDOS.plot(data['DOS'], data['E'], label=labels[1], color='black', marker='o', linestyle='')
        else:  # more than 3 plots
            for index, data in enumerate(data_list):
                for index2, band in enumerate(data['Bandstructures']):
                    if index2 == 0:
                        axBand.plot(band, color=color_list[index], label=labels[index])
                    else:
                        axBand.plot(band, color=color_list[index])
                    axDOS.plot(data['DOS'], data['E'], label=labels[index], color=color_list[index])
        axDOS.set_xlabel('DOS (states/eV)')
        axDOS.set_ylabel('Energy (eV)')
        axDOS.set_title('Density of states')
        axDOS.grid(True)
        axDOS.invert_xaxis()
        if len(set(labels)) > 1:
            axDOS.legend()
        axBand.label_outer()
        axBand.grid(True)
        n_pos_symmetry_points = (data['parameters']['n_pos_symmetry_points'])
        plt.xticks(n_pos_symmetry_points,
                   [f"${symmetry_point}$" for symmetry_point in data['parameters']['labels']])
        axBand.set_title('Band structure: ' + data['material'])
        if 'DOS' not in data.keys():
            axBand.legend()
        axBand.grid(True)
        axBand.set_ylabel('Energy (eV)')

        return

    def plot_data_file(data, variable, color='blue'):
        if 'Ep_range' in data.keys():
            x = data['Ep_range']
        elif 'E' in data.keys():
            x = data['E']
        else:
            x = None

        if variable == 'JDOS':
            normalize = 1  # we normalize the results for the JDOS (as is done with the JDOS for the RPA spectrum)
        else:
            normalize = 0

        if normalize == 1:
            y = data[variable] / (np.sum(data[variable]) * (x[1] - x[0]))
        else:
            y = data[variable]

        if variable != 'Bandstructures':
            plt.plot(x, y, color=color)
        else:
            for band in y:
                plt.plot(band, color=color)
        return

    while len(data_list) < number_of_plots:  # For the remaining plots we compare to the results save in json files
        if len(data_list) == 0 or len(data_list) < number_of_plots - 1:
            path = askopenfilename(initialdir='results')  # For example: old results to which we want to compare
        else:
            path = askopenfilename(initialdir='external_results')
            # For example: compare with results from the RPA spectrum. Manually navigate to results folder if needed
        with open(path) as f:
            data_list.append(json.load(f))

    # TODO: Add support for plotting different materials together.
    if 'material' in data_list[0].keys():
        material = data_list[0]['material']
    else:
        material = os.path.basename(path)

    for index, variable in enumerate(['JDOS', '\u03B5_i', '\u03B5_r', '\u03B1', 'n_ref', 'DOS', 'Bandstructures']):
        if variable == 'DOS' in data_list[0].keys() and 'Bandstructures' in data_list[0].keys():
            plot_DOS_plus_bands(data_list)
        elif variable in data_list[0].keys():
            plt.figure()
            prepare_plot(material, variable)
            for index, data in enumerate(data_list):
                if variable in data.keys():
                    plot_data_file(data, variable, color=color_list[index])
            if len(set(labels)) > 1:
                plt.legend(labels)
    plt.show()
    return


if __name__ == '__main__':
    """Generates figures for files from the results folders"""
    generate_plots(number_of_plots=2)
    # generate_plots(number_of_plots=3, labels=['fitted', 'wannier', 'RPA'], color_list=['b', 'g', 'r'])

    # TODO: Automatically read the symmetry points and the number of points per symmetry line from the wannier90 input
    #  file.
    """Converts external bandstructures to a JSON file that is used by fit_bandstructure.py. JSON files is saved to 
    fitting/fit_data/bandstructures"""
    # # symmetry_lines = [["X", "\\Gamma"], ["\\Gamma", "L"]]
    # # n_line = [30, 30]
    # #  Specify the symmetry lines along which the band structures are calculated below:
    # # symmetry_lines = [["\\Gamma", "X"], ["X", "U"], ["K", "\\Gamma"], ["\\Gamma", "L"], ["L", "W"], ["W", "X"]]
    # # symmetry_lines = [["L", "\\Gamma"], ["\\Gamma", "X"]]
    # # n_line = [50, 59]  # number of points per symmetry line
    # # symmetry_lines = [["L", "\\Gamma"], ["\\Gamma", "X"], ["X", "K"], ["K", "\\Gamma"]]
    # # n_line = [50, 58, 20, 62]  # number of points per symmetry line
    # symmetry_lines = [["L", "\\Gamma"], ["\\Gamma", "X"]]
    # n_line = [30, 36]  # number of points per symmetry line
    # # n_line = [30, 30]  # number of points per symmetry line
    # valence_count = np.inf
    # conductance_count = np.inf
    # external_bands_to_JSON(symmetry_lines, n_line, valence_count, conductance_count)

    """Converts Wannier90 Hamiltonians to usable JSON file. JSON files is saved to fitting/fit_data/matrices"""
    # Wannier_dat_to_JSON()

    """Convert optical properties from RPA spectrum to usable json file"""
    # Optical_properties_to_JSON()

    """Compare results with results from Webplotdigitizer"""
    # Compare_to_Webplotdigitizer(label_external="Chadi & Cohen")
