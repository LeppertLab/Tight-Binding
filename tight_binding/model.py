#!/usr/bin/env python

from sympy import exp, symbols, I, Mul, zeros, diag, Matrix, Expr, sqrt, Symbol, lambdify
from sympy.vector import CoordSys3D, dot
from tight_binding.objects import Orbital, UnitCell, Atom, Neighbour, Spin
from tight_binding.slaterkoster import ParamCollection
import itertools
import sys
import numpy.linalg as LA
import numpy as np

from typing import Set, Dict, Tuple, List, Optional, Union

from tight_binding.spin_orbit import get_soc_matrix

np.set_printoptions(threshold=sys.maxsize)

ORBITAL_CLASSES = ['s', 'p', 'd']

ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "z2"]
}

Number = Union[float, int]
r = CoordSys3D("r")


class Basis():
    def __init__(self):
        self.basis: List[Tuple[Atom, Orbital, Optional[Spin]]] = []

    @classmethod
    def from_unit_cell(cls, unit_cell: UnitCell):
        basis = cls()
        atoms = unit_cell.atoms

        """Basis sorted by orbital position"""
        for atom in atoms:
            for orbital_class, orbitals in atom.active_orbitals.items():
                basis.basis.extend(
                    itertools.product([atom], orbitals, tuple(Spin) if unit_cell.spin_orbit_coupling else (None,)))

        """Basis sorted by orbital type in the order given in the unit cell json file. Not sorted by location."""
        # for atom_type, orbital_type in unit_cell.basis:
        #     for atom in atoms:
        #         for orbital_class, orbitals in atom.active_orbitals.items():
        #             if atom.type == atom_type and orbital_class.name == orbital_type:
        #                 basis.basis.extend(
        #                     itertools.product([atom], orbitals,
        #                                       tuple(Spin) if unit_cell.spin_orbit_coupling else (None,)))
        return basis

    def get_index(self, atom: Atom, orbital: Orbital, spin: Spin) -> int:
        return self.basis.index((atom, orbital, spin))


class TightBinding:
    overlap_integrals: Dict[Tuple[str, str], ParamCollection]
    energies: Dict[Tuple[str, str], Symbol]
    unit_cell: UnitCell

    def __init__(self, unit_cell: UnitCell, r_symbol, order, spin_orbit=False):
        self.unit_cell = unit_cell
        self.basis = Basis.from_unit_cell(unit_cell)
        self.order = order
        self.r = r_symbol

        self.coord_symbols = symbols("x y z", real=True)
        self.angle_symbols = symbols("l m n", real=True)
        self.k_symbols = symbols("kx ky kz", real=True)
        self.k_vector = sequence_to_sympy_vector(self.r, self.k_symbols)

        self.energy_symbols: Set[Symbol] = set()
        self.energy_integral_symbols: Set[Symbol] = set()
        self.soc_symbols: Set[Symbol] = set()

        self.valence_orbitals = {}
        self.conductance_orbitals = {}
        return

    def find_neighbours(self):
        """Finds the neighbours up to the order specified by self.order"""
        all_neighbours = self.unit_cell.sort_neighbours()
        atoms = self.unit_cell.atoms
        neighbours = {}
        for atom in atoms:
            neighbours[atom] = []
        for order in range(self.order):
            ket_atoms = all_neighbours[order][1]
            for ket_atom in ket_atoms:
                neighbours[ket_atom].append(ket_atoms[ket_atom])
        return neighbours

    def construct_hamiltonian(self) -> Matrix:
        """Generates the Hamiltonian"""
        neighbours = self.find_neighbours()

        matrix = zeros(len(self.basis.basis))
        matrix += self._get_tb_matrix(neighbours)

        if self.unit_cell.spin_orbit_coupling:
            matrix += self._get_soc_matrix

        assert matrix.is_hermitian

        return matrix

    def _get_tb_matrix(self, neighbours) -> Matrix:
        """Generates Hamiltonian without SOC components"""
        matrix = zeros(len(self.basis.basis))
        matrix_k0 = zeros(len(self.basis.basis))  # matrix at k = 0. No bloch factor.

        # Diagonal first
        for i, (atom, orbital, spin) in enumerate(self.basis.basis):
            energy_symbol = self.get_energy_symbol(atom, orbital)
            self.energy_symbols.add(energy_symbol)
            matrix[i, i] = energy_symbol
            matrix_k0[i, i] = energy_symbol

        # Fill in all the effect of all neighbouring atoms
        for j, (ket_atom, ket_orbital, ket_spin) in enumerate(
                self.basis.basis):  # For the ket just go through each element in the basis
            for order, bra_atoms_order in enumerate(
                    neighbours[ket_atom]):  # Search for the neighbours of the atom corresponding to the ket orbital
                for bra_atom in bra_atoms_order:  # go through the order of neighbours
                    for orbital_class, bra_orbitals in bra_atom.atom.active_orbitals.items():  # Go through the orbitals for bra
                        for bra_orbital in bra_orbitals:  # Pick one of the orbitals in a specific direction
                            i = self.basis.get_index(bra_atom.atom, bra_orbital,
                                                     ket_spin)  # Gets the index of this neighbour from the predefined basis
                            relative_vector = bra_atom.observed_coordinate - ket_atom.coordinate  # Gets the distance vector between the two atoms
                            bloch_factor = calculate_bloch_factor(relative_vector,
                                                                  self.k_vector)  # Gets the bloch factor (exponential from this distance vector)
                            overlap_integral = self.get_energy_integral(bra_atom, bra_orbital, ket_atom, ket_orbital,
                                                                        order)  # Gets a symbol (parameter) to represent the overlap integral
                            self.energy_integral_symbols.update(overlap_integral.free_symbols)
                            matrix[i, j] += Mul(bloch_factor, overlap_integral)
                            matrix_k0[i, j] += overlap_integral
        return matrix

    @property
    def _get_soc_matrix(self) -> Matrix:
        """Generates the SOC components for the Hamiltonian"""
        block_diagnonal_soc_matrices = []
        for atom in self.unit_cell.atoms:
            orbital_basis: Tuple[Tuple[Orbital, Spin], ...] = tuple()
            for orbital_class, orbitals in atom.active_orbitals.items():
                # Create spin orbit matrix
                spins: Tuple[Spin, ...] = tuple(Spin)
                _orbitals: List[Orbital] = orbitals
                if orbital_class in atom.spin_coupled_orbitals:
                    orbital_basis += tuple(itertools.product(_orbitals, spins))
            if orbital_basis:  # test if it is not empty
                soc_matrix = get_soc_matrix(orbital_basis, atom)
                self.soc_symbols.update(soc_matrix.free_symbols)
                block_diagnonal_soc_matrices.append(soc_matrix)
            else:
                block_diagnonal_soc_matrices.extend([0] * len(orbital_basis))
        return diag(*block_diagnonal_soc_matrices)

    def get_energy_symbol(self, atom: Atom, orbital: Orbital) -> Symbol:
        return Symbol(f"E_{atom.type + orbital.orbital_class.name}", real=True)

    def get_energy_integral(self, bra_atom: Neighbour, bra_orbital: Orbital, ket_atom: Atom, ket_orbital: Orbital,
                            order) -> Expr:
        collection = ParamCollection(bra_atom.atom.type, ket_atom.type, self.coord_symbols, self.angle_symbols)
        energy_integral = collection.get_energy_integral(bra_atom.atom, bra_orbital, ket_atom, ket_orbital, order)

        direction_cosine_vec = self.direction_cosine(ket_atom.coordinate, bra_atom.observed_coordinate)
        direction_cosine_tuple = sympy_unpack(direction_cosine_vec, self.r)

        for i in range(3):
            energy_integral = energy_integral.subs(self.angle_symbols[i], direction_cosine_tuple[i])

        return energy_integral

    @staticmethod
    def direction_cosine(coord1, coord2) -> Expr:
        coord1 = coord1
        coord2 = coord2
        rel_vec = coord2 - coord1
        distance = sqrt(rel_vec.dot(rel_vec))

        output = rel_vec / distance
        return output

    def get_overlap_parameter_symbols(self) -> Set[Symbol]:
        overlap_symbols: Set[Symbol] = set()
        for x in self.overlap_integrals.values():
            overlap_symbols = overlap_symbols.union(set(x.two_center_integrals.values()))

        return overlap_symbols

    def get_energy_parameter_symbols(self) -> Set[Symbol]:
        return set(self.energies.values())

    def get_parameter_symbols(self) -> Set[Symbol]:
        return self.get_energy_parameter_symbols() | self.get_overlap_parameter_symbols() | self.soc_symbols

    def get_lambda_hamiltonian(self, matrix, parameters):
        z = symbols("z")
        for i, j in itertools.product(range(matrix.shape[0]), repeat=2):
            if matrix[i, j] == 0:
                matrix[i, j] = z

        variables = list(itertools.chain(parameters, ['kx', 'ky', 'kz'], [z]))
        lambda_matrix = lambdify(variables, matrix)
        return lambda_matrix

    def get_hamiltonians(self, parameters: Dict[Symbol, Number], k_values):
        matrix = self.construct_hamiltonian()
        substitutions = {}
        for symbol in matrix.free_symbols.difference(self.k_symbols):
            substitutions[symbol] = parameters[str(symbol)]
        lambda_matrix = self.get_lambda_hamiltonian(matrix, substitutions.keys())
        params = np.tile(np.atleast_2d(list(substitutions.values())).T, (1, k_values.shape[1]))
        zeros = np.zeros(k_values.shape[1])
        return lambda_matrix(*params, *k_values, zeros).T

    def get_energy_eigenvalues(self, parameters, k_values):
        """Retrieve energy eigenvalues for a given array of wavevectors."""
        matrices = self.get_hamiltonians(parameters, k_values)
        eigenvalues = LA.eigvalsh(matrices).T
        return eigenvalues

    def get_energy_eigen(self, parameters, k_values):
        """Retrieve both energy eigenfunction and eigenvalues for a given array of wavevectors."""
        matrices = self.get_hamiltonians(parameters, k_values)
        eigenvalues, eigenvectors = LA.eigh(matrices)
        return eigenvalues.T, eigenvectors

    def matrices_R_to_k(self, matrices_R, k_values, r):
        """This function convert the matrices from R to k. Useful when working with the output of wannier90. The only k
        dependence is in the exponentials that we add, thus we can calculate differentiate the Hamiltonian. Returns the
        Hamiltonian in k space, the gradient of the Hamiltonian, the eigenvalues, and the eigenvectors."""
        matrix_shape = np.shape(next(iter(matrices_R.items()))[1])
        matrices_k = np.zeros(matrix_shape + tuple([np.shape(k_values)[1]]), dtype=complex)
        nabla_matrices_k = np.zeros(matrix_shape + tuple([np.shape(k_values)[1]]) + tuple([3]), dtype=complex)
        lattice_vectors = np.array(self.unit_cell.lattice_vectors)
        for R, matrix_R in matrices_R.items():
            R = np.sum(np.array(R) * lattice_vectors).components
            R = [float(R.get(r.i)) if R.get(r.i) is not None else 0.0,
                 float(R.get(r.j)) if R.get(r.j) is not None else 0.0,
                 float(R.get(r.k)) if R.get(r.k) is not None else 0.0]
            k_dot_r = (R * k_values.T).T.sum(axis=0)
            terms = np.exp(1j * k_dot_r) * np.repeat(matrix_R[:, :, np.newaxis], len(k_dot_r), axis=2)
            matrices_k += terms
            nabla_matrices_k += np.repeat(terms[:, :, :, np.newaxis], 3, axis=3) * 1j * R
        input_matrices_r = np.zeros(matrix_shape + tuple([len(matrices_R)]), dtype=complex)
        for index, matrix in enumerate(matrices_R.values()):
            input_matrices_r[:, :, index] += matrix
        input_eigenvalues, input_eigenvectors = LA.eigh(matrices_k.T)
        return matrices_k, nabla_matrices_k, input_eigenvalues.T, input_eigenvectors

    @staticmethod
    def find_conduction_band(eigenvalues):
        """Find the location of the conduction bands and consequently also the valence bands"""
        iscdband = np.zeros(len(eigenvalues[:, 0]), dtype=bool)
        for index, eigenvalues_row in enumerate(eigenvalues):  # It is sufficient to look at the first values only since
            # all of the values inside the band are either positive or negative.
            if np.average(eigenvalues_row) >= 0:  # if true, most if not all of the band is above the Fermi level
                iscdband[index] = 1
        return iscdband

    def find_chosen_bands(self, eigenvalues, nc, nv, Ef, printing=False):
        """Searches for the nc bottommost conduction bands and the nv topmost valence bands."""
        iscdband = self.find_conduction_band(eigenvalues - Ef)
        valence_count = int(np.sum(-(iscdband - 1)))
        conductance_count = int(np.sum(iscdband))
        if printing:
            print('Number of valence bands: ' + str(valence_count))
            print('Number of conduction bands: ' + str(conductance_count))
        return slice(
            valence_count - nv if valence_count - nv > 0 else 0,
            valence_count + nc if valence_count + nc < valence_count + conductance_count else
            valence_count + conductance_count)  # only look at these bands during the calculations

    def shift_eigenvalues(self, eigenvalues, Ef):
        iscdband = self.find_conduction_band(eigenvalues - Ef)
        eigenvalues_v = eigenvalues[np.nonzero(np.round(1 - iscdband))[0], :]
        shift = np.max(eigenvalues_v)
        return eigenvalues - shift  # set the top of the valence band to zero


def sympy_unpack(vector, coordsystem):
    return vector.dot(coordsystem.i), vector.dot(coordsystem.j), vector.dot(coordsystem.k)


def calculate_bloch_factor(r, k) -> Expr:
    return exp(I * dot(k, r))


def all_combinations(*lists):
    return list(itertools.product(*lists))


def sequence_to_sympy_vector(r, array: np.ndarray) -> Expr:
    assert len(array) == 3
    return array[0] * r.i + array[1] * r.j + array[2] * r.k
