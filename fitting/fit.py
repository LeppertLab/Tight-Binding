import numpy as np
from typing import Callable
import numpy.linalg as LA
from scipy.optimize import basinhopping
from math import exp, isclose
import time
import json
from functools import partial

rng = np.random.default_rng()


def symbol_multiplier(symbol, params_dict, adaptive=True):
    """Experimental function.
    This function assigns a multiplier to each of the parameters. This is done because the values for higher order
    neighbors and SOC parameters are expected to be lower. Alternatively this can be disabled by setting adaptive=False.
    Symbol indicates the parameter and params_dict sorts the parameter by one of three types: energy, overlap, and SOC
    """
    if adaptive:
        if symbol in params_dict['energy_symbols']:
            multiplier = 1
        elif symbol in params_dict['soc_symbols']:
            multiplier = 1 / 4
        elif symbol in params_dict['overlap_symbols']:
            symbol = str(symbol)  # convert to str so we can check the last char
            if symbol[-1].isdigit():
                multiplier = 1 / (2 * int(symbol[-1]))  # second nearest neighbor or higher
            else:
                multiplier = 1 / 2  # first nearest neighbor
        else:
            raise Exception('Parameter symbol not recognized')
        return multiplier
    else:
        return 1  # treat all parameters equally


class Basinhopping_funcs:
    # Source basinhopping: https://github.com/scipy/scipy/blob/master/scipy/optimize/_basinhopping.py
    def __init__(self, params, params_dict, stepsize, filename, printing=True,
                 printing_extended=False, target_accept_rate=0.5, adaptive_step_interval=25):
        self.filename = filename
        self.printing = printing
        self.printing_extended = printing_extended

        self.stepsize = stepsize
        self.rng = np.random.default_rng()
        self.parameters = params
        self.params_dict = params_dict
        self.stepsize = stepsize
        self.prev_param_values = []
        self.param_values_lowest_f = []

        self.iteration = 0
        self.local_iteration = 0
        self.unsuccessful_accept = 0
        self.not_moved_count = 0
        self.accept_count = 0  # number of steps that have been accepted
        self.local_accept_count = 0  # number of steps that have been accepted with the new stepsize
        self.accept_rate = 0  # indicates the percentage of steps that have been accepted
        self.local_accept_rate = 0  # indicates the percentage of steps that have been accepted with the new stepsize
        self.lowest_f = np.inf
        self.iteration_lowest_f = 0
        self.prev_f = np.inf

        self.target_accept_rate = target_accept_rate  # local accept rate that you want to target
        self.adaptive_step_interval = adaptive_step_interval  # amount of iterations before step size changes

        self.eigenvalues = None
        self.rng = np.random.default_rng()
        return

    def update_json_file(self, parameters, param_values, f):
        """Stores the parameters from the step with the lowest value of f in a temporary json file. This file can be
        accessed to get the parameters in the case that the code is terminated before it has finished."""
        with open(self.filename, 'r+', encoding='utf8') as file:
            data = json.load(file)
            if f < data["f"]:  # check if the value is lower than the minimums found in all of the other cores.
                file.seek(0)  # start at beginning of file
                data["f"] = f
                for i, parameter in enumerate(parameters):
                    data["parameters"][str(parameter)] = param_values[i]
                json.dump(data, file, ensure_ascii=False)  # write these new values to the json file
                file.truncate()  # delete old data
        return

    def takestep(self, param_values):
        """This function takes a global step in the parameter values. Higher order terms have lower stepsizes."""
        params = self.parameters
        params_dict = self.params_dict
        for index in range(len(param_values)):
            symbol = params[index]
            multiplier = symbol_multiplier(symbol, params_dict, adaptive=True)
            param_values[index] += multiplier * self.rng.uniform(-self.stepsize, self.stepsize)
        return param_values  # Parameters after the step. These parameters have to be minimized with a local minimizer

    def print_output(self, param_values, trial_f, accept):
        """Function that prints details to the console. Calculates some details that give insight into the success
        of the basinhopping function. Also calls the adjust_stepsize function occasionally"""
        if isclose(trial_f, self.prev_f, abs_tol=1e-6):
            if all([isclose(param_values[i], self.prev_param_values[i], abs_tol=1e-4) for i in
                    range(len(param_values))]):
                self.not_moved_count += 1  # Same minimum is reached
                if self.printing_extended:
                    print('Step ' + str(self.iteration) + ': Same minimum is reached')
        if self.iteration > 0:
            if accept:
                self.accept_count += 1
                self.local_accept_count += 1
                if trial_f < self.lowest_f:
                    self.lowest_f = trial_f
                    self.iteration_lowest_f = self.iteration
                    self.param_values_lowest_f = param_values
                    if self.printing:
                        print('New best minimum at step ' + str(self.iteration) + ': f = ' + format(trial_f, ".6f")
                              + '. Parameter values = ' +
                              str(dict(zip(self.parameters, [round(x, 2) for x in param_values]))))
                    self.update_json_file(self.parameters, param_values, trial_f)
                    if self.lowest_f < 1e-6:
                        return True
            self.accept_rate = self.accept_count / self.iteration
            self.local_iteration += 1
            self.local_accept_rate = self.local_accept_count / self.local_iteration
            if self.printing_extended:
                text = 'basinhopping step {iteration}: start_f = {start_f:.6f}, trial_f = {trial_f:.6f}, ' \
                       'lowest_f = {lowest_f:.6f}. accept = {accept}, accept_rate = {acceptance_rate:.3f}, ' \
                       'local_accept_rate = {local_accept_rate:.3f}, stepsize = {stepsize:.3f}, ' \
                       'iteration_lowest_f = {iteration_lowest_f}, amount of iterations not moved = {not_moved_count},' \
                       'parameter values = {parameter_values}'
                print(text.format(iteration=self.iteration, start_f=self.prev_f, trial_f=trial_f,
                                  lowest_f=self.lowest_f, accept=int(accept), acceptance_rate=self.accept_rate,
                                  local_accept_rate=self.local_accept_rate, stepsize=self.stepsize,
                                  iteration_lowest_f=self.iteration_lowest_f, not_moved_count=self.not_moved_count,
                                  parameter_values=str(dict(zip(self.parameters, [round(x, 2) for x in
                                                                                  param_values])))))
            if accept:
                self.prev_f = trial_f
                self.prev_param_values = param_values
        else:
            self.lowest_f = trial_f
            self.prev_f = trial_f
            self.prev_param_values = param_values
            if self.printing:
                text = '\nbasinhopping step {iteration}: f = {f:.3f}'
                print(text.format(iteration=self.iteration, f=trial_f))
        self.iteration += 1
        if self.iteration % self.adaptive_step_interval == 0:
            self.adjust_stepsize()
        return  # Returning True stops the basinhopping algorithm

    def adjust_stepsize(self):
        """This function adjusts the stepsize. If the accept rate is too low our steps are likely to be too high,
        while if the accept rate is too high our steps are likely to be too low. We want to find a balance in our
        acceptance rate (about 0.5) by adjusting the stepsize. In this function the change in stepsize is
        dependent on how far our accept rate is from the target accept rate"""
        old_stepsize = self.stepsize
        self.stepsize = exp(self.local_accept_rate - self.target_accept_rate) * self.stepsize

        # """Default stepsize adjustments"""
        # if self.accept_rate > self.target_accept_rate:
        #     # We're accepting too many steps. This generally means we're
        #     # trapped in a basin. Take bigger steps.
        #     self.stepsize /= 0.9
        # else:
        #     # We're not accepting enough steps. Take smaller steps.
        #     self.stepsize *= 0.9

        if self.printing:
            text = 'Adaptive stepsize (step {iteration}): accept_rate = {accept_rate:.3f}, ' \
                   'local_accept_rate = {local_accept_rate:.3f}, target = {target:.3f}. ' \
                   'new stepsize = {stepsize_new:.3f}, old stepsize = {stepsize_old:.3f}'
            print(text.format(iteration=self.iteration, accept_rate=self.accept_rate,
                              local_accept_rate=self.local_accept_rate, target=self.target_accept_rate,
                              stepsize_new=self.stepsize, stepsize_old=old_stepsize))
        self.local_accept_count = 0
        self.local_accept_rate = 0
        self.local_iteration = 0
        return

    """ Functions that are used to locally minimize the results """

    def fit_func(self, heuristic: Callable, guess_parameters, k_values, matrix_full_lambda, chosen_bands,
                 heuristic_kwargs):
        """Function that is used by the basinhopping function for calculating the eigenvalues, which is used in the other
        functions in order to calculate the value of the heuristics function which is then minimized by basinhopping"""
        params = np.tile(np.atleast_2d(guess_parameters).T, (1, k_values.shape[1]))
        zeros = np.zeros(k_values.shape[1])
        matrices = matrix_full_lambda(*params, *k_values, zeros)
        eigenvalues = LA.eigvalsh(matrices.T).T
        eigenvalues = eigenvalues[chosen_bands, :]
        # self.eigenvalues = eigenvalues
        heuristic_kwargs.update({'eigenvalues': eigenvalues})
        heuristic_kwargs.update({'guess_parameters': guess_parameters})
        return heuristic(**heuristic_kwargs)

    @staticmethod
    def build_heuristic(heuristics):
        """Initializes the function "heuristic_function" such that it recognizes the variable "heuristics" (which contains
        the various functions and their assigned weight) and returns it such that it can be called directly."""

        def heuristic_function(**heuristic_kwargs):
            """Runs each functions and multiply with a predefined weight. Example, fit to the valence bands by setting a
            weight for heuristic_valence_bias while having heuristic_conductance_bias and heuristic_least_squares at zero
            weight."""
            total = 0
            for heuristic, weight in heuristics:
                total += weight * heuristic(**heuristic_kwargs)
            return total

        return heuristic_function

    @staticmethod
    def heuristic_least_squares(input_eigenvalues, eigenvalues, **kwargs):
        """Checks how close the calculated bandstructure is to the input bandstructure"""
        return np.average(np.abs(np.power(eigenvalues - input_eigenvalues, 2)))

    @staticmethod
    def boundary_check(guess_parameters, max_parameter_value=20, **kwargs):
        error = 0
        params_dict = kwargs["params_dict"]
        params = kwargs["params"]
        for index, value in enumerate(guess_parameters):
            symbol = params[index]
            multiplier = symbol_multiplier(symbol, params_dict, adaptive=True)
            if abs(value) > max_parameter_value * multiplier:  # parameters cannot exceed the boundary
                error += abs(value)
        return error


def fit_bandstructure(heuristics, model, matrix, params, k_values,
                      input_eigenvalues, chosen_bands, simulation_parameters,
                      DUMPFILE_NAME, params_dict, initial_energy_parameters=None, initial_overlap_parameters=None):

    matrix_full_lambda = model.get_lambda_hamiltonian(matrix, ([str(x) for x in params]))

    if initial_energy_parameters is None:
        initial_energy_parameters = {}
    if initial_overlap_parameters is None:
        initial_overlap_parameters = {}
    initial_parameter_dictionary = {**initial_energy_parameters, **initial_overlap_parameters}

    initial_parameters = [
        initial_parameter_dictionary[variable]
        if variable in initial_parameter_dictionary
        else rng.uniform(-3, 3)  # random starting values for the overlap parameters
        for variable in params
    ]

    heuristic_kwargs = {
        "params": params,
        "params_dict": params_dict,
        "input_eigenvalues": input_eigenvalues,
        "eigenvalues": np.zeros(np.shape(input_eigenvalues)) if input_eigenvalues is not None else None
    }
    adaptive_step_interval = 25  # determines how often the step size changes
    target_accept_rate = 0.5  # determines the ratio of accepted steps vs total steps
    basin = Basinhopping_funcs(params, params_dict, simulation_parameters.stepsize, DUMPFILE_NAME, printing=True,
                               printing_extended=True, target_accept_rate=target_accept_rate,
                               adaptive_step_interval=adaptive_step_interval)
    partial_fit_func = partial(basin.fit_func, basin.build_heuristic(heuristics))

    print('Starting basinhopping with: temperature = ' + str(simulation_parameters.temperature) +
          ', stepsize = ' + str(simulation_parameters.stepsize) + ', target accept rate = ' + str(target_accept_rate) +
          ', adaptive step interval = ' + str(adaptive_step_interval))
    time1 = time.perf_counter()
    arguments = (k_values, matrix_full_lambda, chosen_bands, heuristic_kwargs)
    fit_result_object = basinhopping(partial_fit_func, initial_parameters, simulation_parameters.basin,
                                     take_step=basin.takestep,
                                     minimizer_kwargs={
                                         "method": "SLSQP",
                                         "args": arguments},
                                     T=simulation_parameters.temperature,
                                     disp=False,
                                     callback=basin.print_output,
                                     niter_success=None)
    time2 = time.perf_counter()
    print("Amount of minimization failures = " + str(fit_result_object.minimization_failures) +
          ". Amount of steps not moved = " + str(basin.not_moved_count) + ". Time elapsed: " + str(time2 - time1) +
          ". lowest_f = " + str(fit_result_object.fun) + ". iteration_lowest_f = " + str(basin.iteration_lowest_f))

    data = {"lowest_f": fit_result_object.fun,
            "temperature": simulation_parameters.temperature,
            "stepsize": simulation_parameters.stepsize,
            "adaptive_step_interval": adaptive_step_interval,
            "iteration_lowest_f": basin.iteration_lowest_f,
            "accept_rate": basin.accept_rate,
            "target_accept_rate": target_accept_rate,
            "minimization_failures": fit_result_object.minimization_failures,
            "not_moved_count": basin.not_moved_count,
            "parameters": [str(x) for x in params],
            "parameter_values": [x for x in fit_result_object.x],
            "elapsed_time": str(time2 - time1)}
    return [data, fit_result_object.x]
