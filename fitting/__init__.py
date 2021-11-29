import json
import numpy as np

from typing import Tuple, Dict, Sequence, cast

Vector3D = Tuple[float, float, float]


class Fitting_data:
    def __init__(self, material: str, Ef=0):
        with open("fitting/fit_data/{}.json".format(material), encoding="utf-8") as f:
            raw_data = json.load(f)

        self.transitions: Dict[
            Tuple[str, str], Tuple[np.ndarray, np.ndarray]
        ] = {}

        self.type = raw_data[0]["type"]
        if self.type == "bandstructures":
            self.transitions = {}
            for segment in raw_data[1:]:
                segment_data = np.array(segment["datapoints"])

                sort_tuple = [
                    (i, np.min(band), np.mean(band)) for i, band in enumerate(segment_data)
                ]
                sorted_tuple = sorted(sort_tuple, key=lambda x: x[2])

                conduction_bands = []
                valance_bands = []
                for i, minimum, mean in sorted_tuple:
                    if minimum > Ef:
                        conduction_bands.append(i)
                    else:
                        valance_bands.append(i)

                direction = cast(Tuple[str, str], tuple(segment["direction"]))

                self.transitions[direction] = (
                    segment_data[valance_bands, :],
                    segment_data[conduction_bands, :],
                )

                self.test_sorting()
        elif self.type == 'wannier':  # fit to wannier matrix elements
            self.matrices = {}
            for Wigner_Seitz_point, matrix in raw_data[1]['matrices'].items():
                self.matrices[eval(Wigner_Seitz_point)] = np.array(matrix[0]) + np.array(matrix[1]) * 1j

    def load_bands(self, transitions: Sequence[str]):
        eigenvalues_list = []
        for transition in self.transitions.keys():
            valance_band = self.transitions[transition][0]
            conduction_band = self.transitions[transition][1]
            valance_band_count = len(valance_band)
            conduction_band_count = len(conduction_band)

            points_per_direction = conduction_band.shape[1]

            eigenvalues = np.empty(
                (valance_band_count + conduction_band_count, points_per_direction)
            )

            eigenvalues[valance_band_count:, :] = conduction_band[
                :conduction_band_count, :
            ]
            eigenvalues[:valance_band_count, :] = valance_band[
                -1 * valance_band_count :, :
            ]

            eigenvalues_list.append(eigenvalues)

        eigenvalues = np.concatenate(eigenvalues_list, 1)

        return eigenvalues

    def get_points_per_direction(self):
        points_per_direction_list = []
        for transition in self.transitions.keys():
            points_per_direction_list.append(self.transitions[transition][0].shape[1])

        return points_per_direction_list

    def get_transitions(self):
        return tuple(self.transitions.keys())

    def get_symmetry_points(self):
        transitions = list(self.transitions.keys())
        if not transitions:
            return None
        symmetry_points = [transitions[0][0]]
        for index in range(len(transitions)):
            if index + 1 < len(transitions):
                if transitions[index][1] == transitions[index + 1][0]:
                    symmetry_points.append(transitions[index][1])
                else:
                    symmetry_points.append(transitions[index][1])
                    symmetry_points.append('jump')  # There is a jump in the symmetry points
                    symmetry_points.append(transitions[index + 1][0])
            else:
                symmetry_points.append(transitions[index][1])
        return symmetry_points

    def test_sorting(self):
        for transition in self.transitions.values():
            for eigenvalues in transition:
                assert np.all(eigenvalues[:-1, :] <= eigenvalues[1:, :])  # Check if the bandstructures are in ascending order

            assert np.all(transition[0][-1, :] <= transition[1][0, :])  # Checks if all valence band values are lower than the conduction band values
