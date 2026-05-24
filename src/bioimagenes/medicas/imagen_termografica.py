"""
La clase ImagenTermografica representa imágenes en las cuales cada
píxel almacena un valor de temperatura. Esta implementación hereda
de la clase Imagen y agrega funcionalidades específicas para el
procesamiento, análisis y visualización de datos térmicos.

La clase permite:

- Interpretar valores térmicos a partir de datos raw.
- Convertir entre distintas unidades de temperatura.
- Detectar puntos calientes.
- Segmentar regiones térmicas.
- Generar mapas de calor.
- Visualizar imágenes termográficas.
- Normalizar intensidades térmicas.
- Registrar operaciones en el historial asociado.

Unidades soportadas
-------------------
- Celsius ("C")
- Kelvin ("K")
- Fahrenheit ("F")

Dependencias
------------
- numpy
- matplotlib
- bioimagenes.core.imagen
- bioimagenes.core.info

Autor
-----
Proyecto Bioimágenes
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


class ImagenTermografica(Imagen):
    """
    Clase especializada para representar imágenes termográficas.

    Cada píxel de la imagen representa una temperatura medida
    o calculada en una unidad determinada.

    Herencia
    --------
    ImagenTermografica hereda de la clase Imagen e incorpora
    atributos y métodos específicos para procesamiento térmico.

    Atributos adicionales
    ---------------------
    unidad : str
        Unidad de temperatura utilizada en la imagen.

        Valores válidos:
        - "C" : Celsius
        - "K" : Kelvin
        - "F" : Fahrenheit

    rango_temp : tuple[float, float]
        Rango térmico representado por la imagen:
        (temperatura_mínima, temperatura_máxima).

    escala : tuple[float, float]
        Parámetros de conversión desde valores raw hacia temperatura:

            temperatura = raw * factor + offset

        donde:
        - factor = escala[0]
        - offset = escala[1]

    umbral_calor : float
        Umbral térmico utilizado para identificar zonas calientes.
    """

    _UNIDADES_VALIDAS = ("C", "K", "F")

    # -------------------------
    # CONSTRUCTOR
    # -------------------------
    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        unidad: str = "C",
        rango_temp: tuple = (20.0, 40.0),
        escala: tuple = (1.0, 0.0),
        umbral_calor: float = 37.0,
    ):
        """
        Inicializa una imagen termográfica.

        Parámetros
        ----------
        data : np.ndarray
            Matriz bidimensional con datos térmicos.

        info : Info
            Objeto con metadatos asociados a la imagen.

        unidad : str, opcional
            Unidad térmica utilizada.
            Por defecto: "C".

        rango_temp : tuple, opcional
            Rango de temperaturas representadas.
            Formato:
                (temperatura_mínima, temperatura_máxima)

        escala : tuple, opcional
            Parámetros de conversión de datos raw.

            Fórmula aplicada:
                temperatura = raw * factor + offset

        umbral_calor : float, opcional
            Temperatura umbral para detección de zonas calientes.

        Excepciones
        -----------
        ValueError
            Si:
            - data no es 2-D.
            - unidad no es válida.
            - rango_temp es inválido.
            - escala es inválida.

        TypeError
            Si umbral_calor no es numérico.
        """

        if data.ndim != 2:
            raise ValueError(
                "ImagenTermografica requiere un array 2-D. "
                f"Se recibió ndim={data.ndim}."
            )

        if unidad not in self._UNIDADES_VALIDAS:
            raise ValueError(f"unidad debe ser una de {self._UNIDADES_VALIDAS}")

        if (
            not isinstance(rango_temp, tuple)
            or len(rango_temp) != 2
            or rango_temp[0] >= rango_temp[1]
        ):
            raise ValueError(
                "rango_temp debe ser una tupla (min, max) con min < max"
            )

        if (
            not isinstance(escala, tuple)
            or len(escala) != 2
            or not all(isinstance(v, (int, float)) for v in escala)
        ):
            raise ValueError(
                "escala debe ser una tupla de dos numéricos (factor, offset)"
            )

        if not isinstance(umbral_calor, (int, float)):
            raise TypeError("umbral_calor debe ser numérico")

        super().__init__(data, info)

        self._unidad = unidad
        self._rango_temp = rango_temp
        self._escala = escala
        self._umbral_calor = float(umbral_calor)

        info["historial"].modificar_historial(
            f"Termografía creada — unidad: {unidad}, umbral_calor: {umbral_calor}"
        )

    # =========================================================
    # PROPIEDADES
    # =========================================================

    @property
    def unidad(self) -> str:
        """
        Retorna la unidad de temperatura utilizada.

        Retorna
        -------
        str
            Unidad térmica ("C", "K" o "F").
        """
        return self._unidad

    @property
    def rango_temp(self) -> tuple:
        """
        Retorna el rango térmico configurado.

        Retorna
        -------
        tuple
            (temperatura_mínima, temperatura_máxima)
        """
        return self._rango_temp

    @property
    def escala(self) -> tuple:
        """
        Retorna la escala utilizada para interpretar temperaturas.

        Retorna
        -------
        tuple
            (factor, offset)
        """
        return self._escala

    @property
    def umbral_calor(self) -> float:
        """
        Retorna el umbral utilizado para detectar zonas calientes.

        Retorna
        -------
        float
            Temperatura umbral.
        """
        return self._umbral_calor

    # =========================================================
    # MÉTODOS ESPECÍFICOS
    # =========================================================

    def interpretar_temperatura(self) -> np.ndarray:
        """
        Convierte los valores raw utilizando la escala configurada.

        Fórmula aplicada
        ----------------
            temperatura = raw * factor + offset

        Retorna
        -------
        np.ndarray
            Matriz con temperaturas interpretadas.

        Notas
        -----
        Este método no modifica self._data.
        """

        factor, offset = self._escala

        temperaturas = self._data.astype(np.float64) * factor + offset

        self._info["historial"].modificar_historial(
            f"Temperatura interpretada (factor={factor}, offset={offset})"
        )

        return temperaturas

    def obtener_rango(self) -> tuple:
        """
        Obtiene el rango térmico actual de la imagen.

        Retorna
        -------
        tuple
            (valor_mínimo, valor_máximo)
        """

        return (float(self._data.min()), float(self._data.max()))

    def convertir_a_temperatura(self, unidad_destino: str) -> "ImagenTermografica":
        """
        Convierte la imagen térmica a otra unidad de temperatura.

        Conversiones soportadas
        -----------------------
        - Celsius ↔ Kelvin
        - Celsius ↔ Fahrenheit
        - Kelvin ↔ Fahrenheit

        Parámetros
        ----------
        unidad_destino : str
            Unidad térmica destino.

        Retorna
        -------
        ImagenTermografica
            Nueva imagen convertida.

        Excepciones
        -----------
        ValueError
            Si la unidad no es válida o coincide con la actual.
        """

        if unidad_destino not in self._UNIDADES_VALIDAS:
            raise ValueError(
                f"unidad_destino debe ser una de {self._UNIDADES_VALIDAS}"
            )

        if unidad_destino == self._unidad:
            raise ValueError("La imagen ya está en la unidad solicitada")

        temp_celsius = self._a_celsius()

        if unidad_destino == "K":
            nueva_data = temp_celsius + 273.15

        elif unidad_destino == "F":
            nueva_data = temp_celsius * 9 / 5 + 32

        else:
            nueva_data = temp_celsius

        nueva_info = Info(
            dimensiones=nueva_data.shape,
            brillo=self._info["brillo"],
            ruta_origen=self._info["ruta_origen"],
        )

        nueva_info["historial"].modificar_historial(
            f"Convertida de {self._unidad} a {unidad_destino}"
        )

        nuevo_rango = (
            float(nueva_data.min()),
            float(nueva_data.max()),
        )

        return ImagenTermografica(
            nueva_data,
            nueva_info,
            unidad=unidad_destino,
            rango_temp=nuevo_rango,
            escala=(1.0, 0.0),
            umbral_calor=self._convertir_umbral(unidad_destino),
        )

    def mapa_calor(self, colormap: str = "hot") -> None:
        """
        Visualiza la imagen como un mapa de calor.

        Parámetros
        ----------
        colormap : str, opcional
            Colormap utilizado para la visualización.
            Por defecto: "hot".
        """

        plt.figure(figsize=(7, 5))

        im = plt.imshow(
            self._data,
            cmap=colormap,
            vmin=self._rango_temp[0],
            vmax=self._rango_temp[1]
        )

        plt.colorbar(im, label=f"Temperatura ({self._unidad})")

        plt.title(
            f"Mapa de calor — umbral caliente: "
            f"{self._umbral_calor} °{self._unidad}"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

        self._info["historial"].modificar_historial(
            f"Mapa de calor visualizado ({colormap})"
        )

    def detectar_puntos_calientes(self, umbral: float = None) -> np.ndarray:
        """
        Detecta regiones cuya temperatura supera un umbral.

        Parámetros
        ----------
        umbral : float, opcional
            Umbral térmico utilizado.
            Si no se especifica, se utiliza self._umbral_calor.

        Retorna
        -------
        np.ndarray
            Máscara booleana 2-D.

            - True  -> píxel caliente
            - False -> píxel frío

        Excepciones
        -----------
        TypeError
            Si el umbral no es numérico.
        """

        if umbral is None:
            umbral = self._umbral_calor

        if not isinstance(umbral, (int, float)):
            raise TypeError("umbral debe ser numérico")

        mascara = self._data > umbral

        cantidad = int(mascara.sum())

        self._info["historial"].modificar_historial(
            f"Puntos calientes detectados "
            f"(umbral={umbral}): {cantidad} píxeles"
        )

        return mascara

    def segmentar_por_umbral(self, umbral: float = None) -> np.ndarray:
        """
        Segmenta la imagen térmica utilizando un umbral.

        Parámetros
        ----------
        umbral : float, opcional
            Umbral utilizado para segmentación.

        Retorna
        -------
        np.ndarray
            Imagen binaria segmentada:

            - 0   -> región fría
            - 255 -> región caliente
        """

        mascara = self.detectar_puntos_calientes(umbral)

        segmentada = np.where(mascara, 255, 0).astype(np.uint8)

        self._info["historial"].modificar_historial(
            "Segmentación por umbral aplicada"
        )

        return segmentada

    def normalizar(self) -> None:
        """
        Normaliza los datos térmicos al rango [0, 1].

        Notas
        -----
        Sobrescribe directamente self._data.
        """

        super().normalizar()

    def visualizar(self, slice_index=None) -> None:
        """
        Visualiza la imagen termográfica.

        Parámetros
        ----------
        slice_index : opcional
            Parámetro heredado para compatibilidad.
        """

        plt.figure(figsize=(6, 5))

        plt.imshow(self._data, cmap="RdBu_r")

        plt.colorbar(label=f"Temperatura ({self._unidad})")

        plt.title(
            f"Termografía — rango: "
            f"{self._rango_temp[0]}–{self._rango_temp[1]} °{self._unidad}"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    # =========================================================
    # AUXILIARES PRIVADOS
    # =========================================================

    def _a_celsius(self) -> np.ndarray:
        """
        Convierte internamente los datos a grados Celsius.

        Retorna
        -------
        np.ndarray
            Datos convertidos a Celsius.
        """

        d = self._data.astype(np.float64)

        if self._unidad == "C":
            return d

        elif self._unidad == "K":
            return d - 273.15

        else:
            return (d - 32) * 5 / 9

    def _convertir_umbral(self, unidad_destino: str) -> float:
        """
        Convierte el umbral térmico a otra unidad.

        Parámetros
        ----------
        unidad_destino : str
            Unidad térmica destino.

        Retorna
        -------
        float
            Umbral convertido.
        """

        celsius = self._umbral_calor

        if self._unidad == "K":
            celsius = self._umbral_calor - 273.15

        elif self._unidad == "F":
            celsius = (self._umbral_calor - 32) * 5 / 9

        if unidad_destino == "K":
            return celsius + 273.15

        elif unidad_destino == "F":
            return celsius * 9 / 5 + 32

        return celsius

    # =========================================================
    # REPRESENTACIÓN
    # =========================================================

    def __str__(self) -> str:
        """
        Retorna una representación textual resumida del objeto.

        Retorna
        -------
        str
            Cadena descriptiva de la imagen termográfica.
        """

        return (
            f"ImagenTermografica(shape={self._data.shape}, "
            f"unidad='{self._unidad}', "
            f"rango={self._rango_temp}, "
            f"umbral_calor={self._umbral_calor})"
        )