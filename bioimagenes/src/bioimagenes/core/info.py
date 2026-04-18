from typing import Any, Tuple
from bioimagenes.core.historial import Historial   # ajustá el import según tu estructura


class Info:

    # -------------------------
    # CONSTRUCTOR
    # -------------------------
    def __init__(
        self,
        dimensiones: Tuple[int, ...],
        brillo: float,
        historial: Historial = None,
        cortada: bool = False
    ):

        # -------------------------
        # VALIDACIÓN DE DIMENSIONES
        # -------------------------
        if not isinstance(dimensiones, tuple):
            raise TypeError("dimensiones debe ser una tupla")

        for d in dimensiones:

            if not isinstance(d, int):
                raise TypeError("cada dimensión debe ser un entero")

            if d <= 0:
                raise ValueError("las dimensiones deben ser positivas")

        # -------------------------
        # VALIDACIÓN DE BRILLO
        # -------------------------
        if not isinstance(brillo, (int, float)):
            raise TypeError("brillo debe ser numérico")

        # -------------------------
        # VALIDACIÓN DE CORTADA
        # -------------------------
        if not isinstance(cortada, bool):
            raise TypeError("cortada debe ser un booleano")

        # -------------------------
        # VALIDACIÓN / CREACIÓN DE HISTORIAL
        # -------------------------
        if historial is None:
            # Si no se pasa historial, se crea uno nuevo
            self._historial = Historial()

        else:
            # Verificamos que sea un objeto Historial
            if not isinstance(historial, Historial):
                raise TypeError("historial debe ser un objeto de tipo Historial")

            self._historial = historial

        # -------------------------
        # ASIGNACIÓN DE ATRIBUTOS
        # -------------------------
        self._dimensiones = dimensiones
        self._brillo = float(brillo)
        self._cortada = cortada

    # -------------------------
    # MÉTODO __contains__
    # -------------------------
    def __contains__(self, key: str) -> bool:

        claves_validas = [
            "dimensiones",
            "brillo",
            "historial",
            "cortada"
        ]

        return key in claves_validas

    # -------------------------
    # MÉTODO __getitem__
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

        else:
            raise KeyError("Clave inválida")

    # -------------------------
    # MÉTODO __str__
    # -------------------------
    def __str__(self) -> str:

        texto = "Info("

        texto += "dimensiones=" + str(self._dimensiones) + ", "
        texto += "brillo=" + str(self._brillo) + ", "
        texto += "cortada=" + str(self._cortada)

        texto += ")"

        return texto