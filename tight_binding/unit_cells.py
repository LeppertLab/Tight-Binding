from .objects import UnitCell, Atom, Orbital

from sympy.vector import CoordSys3D
from fractions import Fraction
from typing import List, Dict

import json

r = CoordSys3D('r')

ORBITAL_CLASSES = ["s", "p", "d"]
ORBITALS = ["s", "px", "py", "pz", "xz", "yz", "xy", "x2-y2", "3z2-r2"]
ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "3z2-r2"]
}


def unit_cell_parser(structure, r: CoordSys3D):
    lattice_vectors = []
    for lattice_vector in structure['lattice']:
        lattice_vectors.append(
            Fraction(lattice_vector[0]) * r.i
            + Fraction(lattice_vector[1]) * r.j
            + Fraction(lattice_vector[2]) * r.k
        )

    brillouin_zone = structure['brillouin_zone']

    unit_cell = UnitCell(lattice_vectors, brillouin_zone)

    for atom in structure['basis']:
        for position in atom['positions']:
            orbitals: Dict[str, List[Orbital]] = {
                'valence': [],
                'conductance': []
            }
            for band in ["valence", "conductance"]:
                for active_orbital_type in atom[band + '_orbitals']:
                    if active_orbital_type in ORBITAL_MAP:
                        orbitals[band].extend(Orbital.from_orbital_class(active_orbital_type))
                    elif active_orbital_type in ORBITALS:
                        orbitals[band].append(Orbital(active_orbital_type))
                    else:
                        raise ValueError("Active orbital: {:s} does not exist".format(active_orbital_type))

            flip_odd = True if 'flip_odd_orbitals' in atom and atom['flip_odd_orbitals'] else False

            atom_instance = Atom(
                atom["type"],
                tuple_to_vector(position, r),
                tuple(orbitals['valence']),
                tuple(orbitals['conductance']),
                flip_odd
            )
            unit_cell.add_atom(atom_instance)

    return unit_cell


def load_unit_cell(filename, r):
    with open(filename, encoding="utf-8") as json_file:
        return unit_cell_parser(json.load(json_file), r)


def tuple_to_vector(tuple, r: CoordSys3D):
    return Fraction(tuple[0]) * r.i + Fraction(tuple[1]) * r.j + Fraction(tuple[2]) * r.k
