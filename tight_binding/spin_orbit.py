from sympy import Symbol, sqrt, Expr, I
from itertools import product

import sympy.matrices as matrices

from collections import OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction

from tight_binding.objects import Spin, Atom, Orbital, OrbitalClass

from typing import List, Tuple, Sequence, Dict, Union
Number = Union[int, float]
Harmonic_Map = Dict[str, List[Tuple[Union[Expr, Number], Tuple[int, int]]]]

"""The following orbital maps can be found on: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics. Scroll to the 
section about real spherical harmonics. Here the cartesian orbitals are related to the spherical harmonics in the way 
given below. Follow the above link to expand the model for higher values of the azimuthal quantum number l. If you do
this also update the function get_soc_matrix to allow higher values for ml."""

s_orbital_harmonic_map: Harmonic_Map = OrderedDict([
    ("s", [
        (1, (0, 0))  # prefactor, l, m
    ])
])

p_orbital_harmonic_map: Harmonic_Map = OrderedDict([
    ("px", [
        (1/sqrt(2), (1, -1)),
        (-1/sqrt(2), (1, 1))
    ]),
    ("py", [
        (I/sqrt(2), (1, -1)),
        (I/sqrt(2), (1, 1))
    ]),
    ("pz", [
        (1, (1, 0)),
    ]),
])

d_orbital_harmonic_map: Harmonic_Map = OrderedDict([
    ("xy", [
        (I/sqrt(2), (2, -2)),
        (-I/sqrt(2), (2, 2))
    ]),
    ("xz", [
        (1/sqrt(2), (2, -1)),
        (-1/sqrt(2), (2, 1))
    ]),
    ("yz", [
        (I/sqrt(2), (2, -1)),
        (I/sqrt(2), (2, 1))
    ]),
    ("x2-y2", [
        (1/sqrt(2), (2, -2)),
        (1/sqrt(2), (2, 2))
    ]),
    ("z2", [
        (1, (2, 0)),
    ]),
])

orbital_harmonic_map = OrderedDict()
orbital_harmonic_map.update(s_orbital_harmonic_map)
orbital_harmonic_map.update(p_orbital_harmonic_map)
orbital_harmonic_map.update(d_orbital_harmonic_map)


@dataclass
class State:
    """A combined angular momentum/spin state. Below in the functions are the operators which can be applied to this
    state, which can alter the state in case of the raising/lowering operators. The factor hbar^2 is taken to be inside
    the SOC parameter inside the tight-binding model."""

    l: int
    ml: int
    s: Fraction
    ms: Fraction

    def Lp(self):
        """Apply the angular momentum plus operator (L+) to the state and return the factor."""
        factor = sqrt(self.l * (self.l + 1) - self.ml * (self.ml + 1))
        self.ml += 1
        return factor

    def Lm(self):  # Apply the L- operator to the state
        """Apply the angular momentum minus operator (L-) to the state and return the factor."""
        factor = sqrt(self.l * (self.l + 1) - self.ml * (self.ml - 1))
        self.ml -= 1
        return factor

    def Lz(self):  # Apply the Lz operator to the state
        """Apply the angular momentum z operator (Lz) to the state and return the factor."""
        return self.ml

    def Sp(self):  # Apply the S+ operator to the state
        """Apply the spin plus operator (S+) to the state and return the factor."""
        factor = sqrt(self.s * (self.s + 1) - self.ms * (self.ms + 1))
        self.ms += 1
        return factor

    def Sm(self):  # Apply the S- operator to the state
        """Apply the spin minus operator (S-) to the state and return the factor."""
        factor = sqrt(self.s * (self.s + 1) - self.ms * (self.ms - 1))
        self.ms -= 1
        return factor

    def Sz(self):  # Apply the Sz operator to the state
        """Apply the spin z operator (Sz) to the state and return the factor."""
        return self.ms

    def clone(self):
        """Shallow clone the State. Clones the state so that we have the state before and after applying the above
        operators."""
        return replace(self)

    def __eq__(s, o):
        """Compare if the state is equal."""
        return s.l == o.l and s.ml == o.ml and s.s == o.s and s.ms == o.ms

    def is_legal(self):
        """Check if a state abides by the laws of physics."""
        return abs(self.ml) <= self.l and abs(self.ms) <= self.s


def get_soc_basis(orbital_basis: Sequence[Tuple[Orbital, Spin]], atom: Atom):
    """This function indicates the basis of the spin orbit Hamiltonian (basis of spherical harmonics)"""
    # Check if the basis is unique
    assert len(orbital_basis) == len(set(orbital_basis))

    l_values = [orbital_class.value for orbital_class in atom.spin_coupled_orbitals]

    states_content = list(
        product(
            l_values,
            [-2, -1, 0, 1, 2],
            [Fraction(1 / 2)],
            [Fraction(1 / 2), Fraction(-1 / 2)],
        )
    )

    states = [State(*state) for state in states_content if abs(state[1]) <= state[0]]

    return states


def get_harmonic_soc_matrix(soc_basis, atom: Atom):

    lpsm_matrix = matrices.zeros(len(soc_basis))
    lmsp_matrix = matrices.zeros(len(soc_basis))
    lzsz_matrix = matrices.zeros(len(soc_basis))

    lambdas = {
        orbital_class.value:
        Symbol('lambda_{atom}_{orbital_class}'.format(atom=atom.type, orbital_class=orbital_class.name), real=True)
        for orbital_class in OrbitalClass
    }

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lp()
        factor *= st.Sm()

        if st.is_legal():
            m = soc_basis.index(st)
            lpsm_matrix[n, m] = factor * lambdas[st.l]  # L+S- part of H_soc

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lm()
        factor *= st.Sp()

        if st.is_legal():
            m = soc_basis.index(st)
            lmsp_matrix[n, m] = factor * lambdas[st.l]  # L-S+ part of H_soc

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lz()
        factor *= st.Sz()

        if st.is_legal():
            m = soc_basis.index(st)
            lzsz_matrix[n, m] = factor * lambdas[st.l]  # LzSz part of H_soc

    H_soc = lzsz_matrix + Fraction(1/2) * (lpsm_matrix + lmsp_matrix)

    return H_soc


def get_soc_matrix(orbital_basis: Sequence[Tuple[Orbital, Spin]], atom: Atom):
    for orbital_class in atom.spin_coupled_orbitals:
        needed_basis_elements = set(product(Orbital.from_orbital_class(orbital_class), Spin))
        assert needed_basis_elements.issubset(orbital_basis)

    soc_basis = get_soc_basis(orbital_basis, atom)
    H_soc = get_harmonic_soc_matrix(soc_basis, atom)  # Spin orbit Hamiltonian in basis of spherical harmonics

    T = matrices.zeros(len(orbital_basis), len(soc_basis))  # Transformation matrix to go from l & ml to cartesian
    for m, (orbital, spin) in enumerate(orbital_basis):
        for factor, harmonic_content in orbital_harmonic_map[str(orbital)]:
            harmonic = State(*harmonic_content, Fraction(1/2), spin.value)
            n = soc_basis.index(harmonic)
            T[m, n] = factor.conjugate()

    return T * H_soc * T.inv()  # Spin orbit Hamiltonian is converted to the basis of the cartesian orbitals
