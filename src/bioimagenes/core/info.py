from typing import Any, Tuple
# #from bioimagenes.core.historial import Historial   # ajustá el import según tu estructura
# import sys
# import os

# sys.path.append(
#     os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "../../")
#     )
# )

import numpy as np
import matplotlib.pyplot as plt
from bioimagenes.core.historial import Historial


class Info:

    # -------------------------
    # CONSTRUCTOR
    # -------------------------
    def __init__(
        self,
        dimensiones: Tuple[int, ...],
        brillo: float,
        historial: Historial = None,
        cortada: bool = False,
        ruta_origen: str = None   
    ):

        # -------------------------
        # VALIDACIÓN DIMENSIONES
        # -------------------------
        if not isinstance(dimensiones, tuple):
            raise TypeError("dimensiones debe ser una tupla")

        for d in dimensiones:
            if not isinstance(d, int):
                raise TypeError("cada dimensión debe ser un entero")
            if d <= 0:
                raise ValueError("las dimensiones deben ser positivas")

        # -------------------------
        # VALIDACIÓN BRILLO
        # -------------------------
        if not isinstance(brillo, (int, float)):
            raise TypeError("brillo debe ser numérico")

        # -------------------------
        # VALIDACIÓN CORTADA
        # -------------------------
        if not isinstance(cortada, bool):
            raise TypeError("cortada debe ser booleano")

        # -------------------------
        # VALIDACIÓN HISTORIAL
        # -------------------------
        if historial is None:
            self._historial = Historial()
        else:
            if not isinstance(historial, Historial):
                raise TypeError("historial debe ser tipo Historial")
            self._historial = historial

        # -------------------------
        # NUEVO: VALIDACIÓN RUTA
        # -------------------------
        if ruta_origen is not None and not isinstance(ruta_origen, str):
            raise TypeError("ruta_origen debe ser string o None")

        # -------------------------
        # ASIGNACIÓN
        # -------------------------
        self._dimensiones = dimensiones
        self._brillo = float(brillo)
        self._cortada = cortada
        self._ruta_origen = ruta_origen   

    # -------------------------
    # CONTAINS
    # -------------------------
    def __contains__(self, key: str) -> bool:

        claves = [
            "dimensiones",
            "brillo",
            "historial",
            "cortada",
            "ruta_origen"   
        ]

        return key in claves

    # -------------------------
    # GETITEM
    # -------------------------
    def __getitem__(self, key: str) -> Any:

        if key == "dimensiones":
            return self._dimensiones

        elif key == "brillo":
            return self._brillo

        elif key == "historial":
            return self._historial

        elif key == "cortada":
            return self._cortada

        elif key == "ruta_origen":   
            return self._ruta_origen

        else:
            raise KeyError("Clave inválida")

    # -------------------------
    # STR
    # -------------------------
    def __str__(self):

        texto = "Info("
        texto += f"dimensiones={self._dimensiones}, "
        texto += f"brillo={self._brillo}, "
        texto += f"cortada={self._cortada}, "
        texto += f"ruta_origen={self._ruta_origen}"
        texto += ")"

        return texto