"""A collection of Tight Binding Parameters."""

import json
from dataclasses import dataclass

from tight_binding.objects import Orbital

from typing import Dict, Tuple


@dataclass
class ParameterCollection:
    """A collection of Tight Binding parameters."""

    energies: Dict[str, float]
    overlap_integrals: Dict[str, float]
    soc_coefficients: Dict[str, float]

    @classmethod
    def from_file(cls, filename: str, order: str):
        with open(filename, encoding="utf-8") as f:
            data = json.loads(f.read())
            if order in data["order"]:
                parameters = data["order"][order]
            else:
                raise NotImplementedError("No parameters found for the given order of neighbours. Check the relevant"
                                          " JSON file located in the tight_binding_parameters folder")
            energies = {
                "E_" + key: value for key, value in parameters["energies"].items()
            }

            overlap_integrals = {
                key: value for key, value in parameters["overlap_integrals"].items()
            }

            if "soc_coefficients" in parameters:
                soc_coefficients = {
                    key: value for key, value in parameters["soc_coefficients" ].items()
                }
            else:
                soc_coefficients = {}

            return cls(energies, overlap_integrals, soc_coefficients)
