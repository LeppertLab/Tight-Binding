Below is a short description of all runnable .py files. See the help section for each of the arguments for more detail into on what they do.

gen_bandstructure.py:
	Generates the band structures for the given material and k-path. Results are saved in the results folder.
fit_bandstructure.py:
	Fits to external eigenvalues that are located in the fitting/fit_data/bandstructures folder. Alternatively, we fit to the eigenvalues that
	we generate from the Wannier Hamiltonians which are located in the fitting/fit_data/matrices folder. Results are saved in the simulations
	folder. One folder is created at the end of the code and during the code the fit_lowest_f JSON file gets updated when a new lowest minimum
	is found.
save_fit.py:
	Opens the folder that we save with fit_bandstructure.py and generates a JSON file at the tight_binding_parameters folder with the parameter
	values.
density_of_states.py:
	Calculates the density of states for the given material and k grid. Results are saved in the results folder.
dielectric_function.py:
	Calculates the optical properties for the given material and k grid. Results are saved in the results folder.
transition_matrix_elemenets.py:
	Short bit of code that I used to calculate the (dipole) transition matrix elements.
data_processing.py:
	Contains various functions that I used for all of the above python scripts.



Below is a short description for each of the folders.

results:
	In this folder the results from most of the python scripts are saved in the form of JSON files.
simulations:
	In this folder the fits from fit_bandstructure.py are saved.
external_results:
	Folder where I store the JSON files for external results (for example the RPA spectrum) to not get them mixed with results from the model.
parameters_guesses:
	This folder can be used to set the starting guesses for each parameter in fit_bandstructure.py. We use random starting guesses if the 	material is not found in this folder.
tight_binding:
	Contains the functions that we use for most of the python scripts. slaterkoster.py contains the expression for the two-center integrals, 	which need to be changed to the more general energy integrals to improve the model.
tight_binding_parameters:
	Contains the parameter values that we use in our models for each material.
unit_cells:
	Contains information about the unit cells for each material.
visualisation:
	Contains functions for plotting the band structures and error plots at the end of gen_bandstructure.py.



