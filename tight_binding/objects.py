"""Collection of objects for a Tight Binding model."""
from dataclasses import dataclass
import numpy as np
from sympy import sqrt
from sympy.vector import CoordSys3D
from itertools import product, chain
import json
from collections import defaultdict
from enum import Enum
import sys

from typing import Tuple, List, Dict, Iterable, Union, Sequence

Number = Union[float, int]

r = CoordSys3D('r')

ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "z2"]
}


class Spin(Enum):
    """Quantum mechanical Spin half enumeration."""

    UP = 1 / 2
    DOWN = -1 / 2


class OrbitalClass(Enum):
    """Orbital Class enumeration."""

    s = 0
    p = 1
    d = 2
    f = 3

    @classmethod
    def has_letter(cls, letter: str):
        """Check if enumeration has a certain name."""
        return letter in cls.__members__


@dataclass(frozen=True, eq=True)
class Orbital:
    """Quantum mechanical orbital."""

    orbital_map = {
        OrbitalClass.s: ["s"],
        OrbitalClass.p: ["px", "py", "pz"],
        OrbitalClass.d: ["xy", "xz", "yz", "x2-y2", "z2"]
    }

    representation: str

    @classmethod
    def from_orbital_class(cls, orbital_class: OrbitalClass):
        """Generate List of Orbitals from a certain azimuthal quantum number."""
        orbitals = []
        for orbital in cls.orbital_map[orbital_class]:
            orbitals.append(cls(orbital))

        return orbitals

    @property
    def parity(self):
        """Partity of the Orbital."""
        return self.azimuthal_quantum_number % 2

    @property
    def azimuthal_quantum_number(self) -> int:
        """Azimuthal quantum number of the Orbital."""
        for i, (orbital_class, orbitals) in enumerate(self.orbital_map.items()):
            if self.representation in orbitals:
                return i

        raise Exception("Orbital has no known quantum number")

    @property
    def l(self):
        """Azimuthal quantum number of the Orbital."""
        return self.azimuthal_quantum_number

    @property
    def orbital_class(self) -> OrbitalClass:
        """Get historical letter representing the azimuthal quantum number."""
        return OrbitalClass(self.azimuthal_quantum_number)

    def __str__(self):
        """Return the geometric representation of the orbital."""
        return self.representation


@dataclass(frozen=True)
class Atom:
    """Object representation of an Atom."""

    type: str
    coordinate: np.ndarray
    orbitals: Tuple[Tuple[OrbitalClass, Tuple[Orbital, ...]], ...]
    spin_coupled_orbitals: Tuple[OrbitalClass, ...]
    flipped_odd_orbitals: bool = False

    @property
    def active_orbitals(self) -> Dict[OrbitalClass, List[Orbital]]:
        """Creates a dictionary which indicates which orbital types are enabled for a certain atom. Also lists the
         different directions of that orbital. For example p has three directions: px, py, pz"""
        all_orbitals: Dict[OrbitalClass, List[Orbital]] = defaultdict(list)

        for orbital_class, orbitals in self.orbitals:
            all_orbitals[orbital_class].extend(orbitals)
        return all_orbitals


# TODO: Replace with AtomObservation that inherits Atom
@dataclass
class Neighbour:
    """An observation of an Atom at a certain location."""

    atom: Atom
    observed_coordinate: np.ndarray


class UnitCell:
    """Unit Cell object."""

    lattice_vectors: np.ndarray
    atom_map: Dict[int, Atom]
    id_map: Dict[Atom, int]
    spin_orbit_coupling: bool = False
    orbital_count: int = 0

    def __init__(self, lattice_vectors, brillouin_zone=None, shape=None):
        """Create a Unit Cell object."""
        self.lattice_vectors = lattice_vectors
        self.int = 0
        self.atom_types = []
        self.atoms: List[Atom] = []
        self.brillouin_zone = brillouin_zone
        self.shape = shape
        self.orbital_map = {}
        self.basis = []

    @classmethod
    def from_dict(cls, structure, r: CoordSys3D):
        """Initialise Unit Cell from dict."""
        lattice_vectors = []
        for lattice_vector in structure['lattice']:
            lattice_vectors.append(
                lattice_vector[0] * r.i
                + lattice_vector[1] * r.j
                + lattice_vector[2] * r.k
            )

        shape = structure['shape']
        brillouin_zone = structure['brillouin_zone']

        unit_cell = cls(lattice_vectors, brillouin_zone, shape)

        ORBITALS = list(chain.from_iterable(ORBITAL_MAP.values()))
        basis = []

        for atom in structure['basis']:
            unit_cell.atom_types.append(atom['type'])
            for position in atom['positions']:
                sorted_orbitals: Dict[OrbitalClass, List[Orbital]] = \
                    {orbital_class: list() for orbital_class in OrbitalClass}
                for active_orbital_type in atom['orbitals']:
                    if OrbitalClass.has_letter(active_orbital_type):
                        orbital_class = OrbitalClass[active_orbital_type]
                        orbitals = Orbital.from_orbital_class(orbital_class)
                        unit_cell.orbital_count += len(orbitals)
                        sorted_orbitals[orbital_class].extend(orbitals)
                        unit_cell.orbital_map[orbital_class] = orbitals
                        basis.append([atom['type'], active_orbital_type])
                    elif active_orbital_type in ORBITALS:
                        orbital = Orbital(active_orbital_type)
                        sorted_orbitals[orbital.orbital_class].append(orbital)
                        unit_cell.orbital_count += 1
                    else:
                        raise ValueError("Active orbital: {:s} does not exist".format(active_orbital_type))

                flip_odd = True if 'flip_odd_orbitals' in atom and atom['flip_odd_orbitals'] else False

                spin_coupled_orbitals = tuple((
                    OrbitalClass[orbital_class_letter]
                    for orbital_class_letter in atom['spin_coupled_orbitals']
                )) \
                    if 'spin_coupled_orbitals' in atom \
                    else ()

                if len(spin_coupled_orbitals) > 0:
                    unit_cell.spin_orbit_coupling = True

                if atom["cartesian"]:  # locations are given in cartesian coordinates
                    location = tuple_to_vector(position, r)
                else:  # locations are given in the basis of the lattice vectors.
                    location = np.sum(np.array(position)*np.array(lattice_vectors))
                atom_instance = Atom(
                    atom["type"],
                    location,
                    tuple((k, tuple(v)) for k, v in sorted_orbitals.items()),
                    spin_coupled_orbitals,
                    flip_odd
                )
                unit_cell.add_atom(atom_instance)
        [unit_cell.basis.append(x) for x in basis if x not in unit_cell.basis]
        return unit_cell

    @classmethod
    def from_file(cls, filename, r):
        """Load unit cell from file."""
        with open(filename, encoding="utf-8") as f:
            return cls.from_dict(json.loads(f.read()), r)

    def add_atom(self, atom: Atom):
        """Add atom in the unit cell."""
        self.atoms.append(atom)

    def add_atoms(self, atoms: Iterable[Atom]):
        """Add multiple atoms to the unit cell."""
        for atom in atoms:
            self.add_atom(atom)

    def get_symmetry_points(self, symmetry_points: Iterable[str]):
        """Get list of symmetry points."""
        symmetry_point_letters = reformat_symmetry_points(symmetry_points)
        return np.array([np.array(self.brillouin_zone[letter]) * 2 * np.pi for letter in symmetry_point_letters])

    def get_k_values(self, number_of_k_values: int, chosen_symmetry_points, n=None, adaptive=1):
        """Generates the k values between the assigned symmetry points. If adaptive is set to 1 then the amount of
        k values scales linearly with the distance between the symmetry points.
        Total amount of k values = number_of_k_values"""
        try:
            positions = self.get_symmetry_points(chosen_symmetry_points)
        except KeyError as e:
            print(f'Symmetry point "{e.args[0]}" was not found for this unit cell', file=sys.stderr, )
            sys.exit(1)

        distances = []
        index_jump = []
        for i, symmetry_point in enumerate(chosen_symmetry_points):
            if ',' in symmetry_point:  # we use a comma to indicate a jump between symmetry points in the k path
                index_jump = index_jump + [i + 1 + len(index_jump)]
        positions = np.array(positions)

        if n is None:
            if adaptive:  # number of points between symmetry points is based on distance
                for i in range(0, len(positions) - 1):
                    if i + 1 in index_jump:  # Position of the jump.
                        distances = distances + [
                            np.array([0, 0, 0])]  # Zero array is added to get zero k-values in this range
                    else:
                        distances = distances + [positions[i + 1] - positions[i]]
                distances = np.array(distances)  # distance vector connecting the consecutive symmetry points
                norms = np.array([np.linalg.norm(distances[i, :]) for i in range(0, len(distances))])  # norm of distances
                n = np.floor(np.cumsum(norms * number_of_k_values / (sum(norms))))  # assign number of indices dependent on the norms
                n = np.insert(n, 0, 0).astype(int)  # insert zero point as reference and convert to int
            else:  # number of points between symmetry points is constant
                n = [0]  # first symmetry point is at index 0
                for i in range(0, len(positions) - 1):
                    if i + 1 in index_jump:
                        n = n + [0]
                    else:
                        n = n + [number_of_k_values / (len(chosen_symmetry_points)-1)]
                n = np.round(np.cumsum(np.array(n))).astype(int)

        # Create the grid with the assigned number of points from the section above
        k = np.zeros([3, n[-1]])  # preallocate the k_values
        for i in range(1, len(positions)):
            if i in index_jump:
                pass
            else:
                k_values_point_to_point = np.linspace(positions[i - 1], positions[i], n[i] - n[i - 1], axis=1)
                k[:, n[i - 1]:n[i]] = k_values_point_to_point
        n = list(dict.fromkeys(n))  # Removes duplicate entries from list
        n = [int(x) for x in n]  # convert intc to int
        return k, n

    def k_values_ibz(self, nk):
        """Gets the k values of the Irreducible Brillouin Zone (IBZ)"""
        iteration = 0
        if self.shape == 'fcc':  # face centered cubic
            if nk % 2 == 0:
                k_range = np.linspace(0, 2 * np.pi, nk + 1)  # assures that we have a k point with value pi
            else:
                k_range = np.linspace(0, 2 * np.pi, nk)
            k = np.zeros([3, int((nk ** 3) / 2)])
            for ny, ky in enumerate(k_range):
                for nx, kx in enumerate(k_range[:ny + 1]):
                    for nz, kz in enumerate(k_range[:nx + 1]):
                        if kx + ky + kz <= 3 * np.pi:
                            k[:, iteration] = [kx, ky, kz]
                            iteration += 1
            k = np.take(k, np.arange(0, iteration), 1)

        elif self.shape == 'sc':  # simple cubic
            k_range = np.linspace(0, np.pi, nk)
            k = np.zeros([3, int(nk ** 3 / 4)])
            for ny, ky in enumerate(k_range):
                for nx, kx in enumerate(k_range[:ny + 1]):
                    for nz, kz in enumerate(k_range[:nx + 1]):
                        k[:, iteration] = [kx, ky, kz]
                        iteration += 1
            k = np.take(k, np.arange(0, iteration), 1)

        else:
            raise NotImplementedError('Could not recognise the shape of the reciprocal lattice.')

        return k

    def sort_neighbours(self) -> List[Tuple[float, Dict[Atom, List[Neighbour]]]]:
        """Create a list of neighbouring pairs sorted and and collected by distance."""
        neighbour_pairs: Dict[float, Dict[Atom, List[Neighbour]]] = defaultdict(lambda: defaultdict(list))

        atoms = self.atoms
        for i, atom in enumerate(atoms):
            for lattice_vector in product([-1, 0, 1], repeat=3):
                for possible_neighbour in atoms:

                    # TODO: use += and don't use ii
                    new_coord = possible_neighbour.coordinate
                    for ii in range(3):
                        new_coord = new_coord + self.lattice_vectors[ii] * lattice_vector[ii]

                    relative_coordinate = new_coord - atom.coordinate

                    distance = sqrt(relative_coordinate.dot(relative_coordinate))

                    if distance == 0:
                        continue

                    neighbour_pairs[distance][atom].append(Neighbour(possible_neighbour, new_coord))

        sorted_neighbourpairs = []
        distances: Iterable[float] = sorted(neighbour_pairs.keys())
        for distance in distances:
            sorted_neighbourpairs.append((distance, neighbour_pairs[distance]))

        return sorted_neighbourpairs


def reformat_symmetry_points(symmetry_points_argument):
    """Function that reformats the symmetry points such that the code can handle a jump between these symmetry points.
    This jump is indicated by a comma between two symmetry points, for example: U,K"""
    symmetry_point_letters = []
    for index, symmetry_point_letter in enumerate(symmetry_points_argument):
        if ',' in symmetry_point_letter:
            location = symmetry_point_letter.find(',')
            symmetry_point_letters = symmetry_point_letters + [symmetry_point_letter[:location]] + \
                                     [symmetry_point_letter[location + 1:]]  # used to calculate k values
        else:
            symmetry_point_letters = symmetry_point_letters + [symmetry_point_letter]
    return symmetry_point_letters


def tuple_to_vector(tuple: Sequence[Number], r: CoordSys3D):
    """Convert sequence of length 3 to 3D vector."""
    # return Fraction(tuple[0]) * r.i + Fraction(tuple[1]) * r.j + Fraction(tuple[2]) * r.k
    return tuple[0] * r.i + tuple[1] * r.j + tuple[2] * r.k
