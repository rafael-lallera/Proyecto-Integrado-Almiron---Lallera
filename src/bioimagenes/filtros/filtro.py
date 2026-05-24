"""
Este módulo implementa una jerarquía de clases para aplicar filtros
espaciales sobre imágenes bidimensionales y multicanal utilizando
operaciones de convolución y filtrado estadístico.

Los filtros implementados permiten realizar tareas clásicas de
procesamiento digital de imágenes, tales como:

- Suavizado.
- Reducción de ruido.
- Filtrado gaussiano.
- Filtrado por mediana.
- Realce de nitidez.

La clase base Filtro proporciona la estructura común para todos los
filtros y define la lógica general de aplicación sobre imágenes
2-D y 3-D.

Clases incluidas
----------------
- Filtro
- FiltroGaussiano
- FiltroMediana
- FiltroSuavizado
- FiltroNitidez

Dependencias
------------
- numpy
- scipy.signal.convolve2d
- scipy.ndimage.median_filter

Autor
-----
Proyecto Bioimágenes
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


class Filtro:
    """
    Clase base para filtros espaciales.

    Esta clase implementa la estructura general necesaria para
    aplicar filtros sobre imágenes utilizando convolución 2-D.

    Características
    ----------------
    - Soporta imágenes 2-D y 3-D.
    - Permite trabajar con distintos kernels.
    - Implementa convolución espacial.
    - Mantiene encapsulados los parámetros del filtro.

    Atributos
    ---------
    _tipo : str
        Nombre identificador del filtro.

    _tamaño : int
        Tamaño del kernel utilizado.

    _kernel : np.ndarray
        Matriz de convolución asociada al filtro.
    """

    def __init__(self, tipo, tamaño=3):
        """
        Inicializa un filtro espacial.

        Parámetros
        ----------
        tipo : str
            Nombre del filtro.

        tamaño : int, opcional
            Tamaño del kernel.
            Debe ser impar y positivo.

        Excepciones
        -----------
        TypeError
            Si tipo no es un string válido.

        ValueError
            Si tamaño no es un entero impar positivo.
        """

        if not isinstance(tipo, str) or not tipo.strip():
            raise TypeError("tipo debe ser un string no vacío")

        if not isinstance(tamaño, int) or tamaño < 1 or tamaño % 2 == 0:
            raise ValueError("tamaño debe ser un entero impar positivo")

        self._tipo = tipo
        self._tamaño = tamaño
        self._kernel = self._construir_kernel()

    def _construir_kernel(self):
        """
        Construye el kernel base del filtro.

        Retorna
        -------
        np.ndarray
            Kernel identidad centrado.

        Notas
        -----
        Este método está diseñado para ser sobrescrito por
        subclases especializadas.
        """

        k = np.zeros((self._tamaño, self._tamaño), dtype=np.float64)

        centro = self._tamaño // 2

        k[centro, centro] = 1.0

        return k

    def _convolucion(self, canal):
        """
        Aplica convolución espacial sobre un canal de imagen.

        Parámetros
        ----------
        canal : np.ndarray
            Canal bidimensional de la imagen.

        Retorna
        -------
        np.ndarray
            Canal filtrado.

        Notas
        -----
        La convolución utiliza:
        - mode="same"
        - boundary="symm"
        """

        salida = convolve2d(
            canal.astype(np.float64),
            self._kernel,
            mode="same",
            boundary="symm"
        )

        return np.clip(salida, 0, 255).astype(canal.dtype)

    def aplicar_filtro(self, data):
        """
        Aplica el filtro sobre una imagen.

        Parámetros
        ----------
        data : np.ndarray
            Imagen de entrada.

            Puede ser:
            - 2-D -> escala de grises
            - 3-D -> imagen multicanal

        Retorna
        -------
        np.ndarray
            Imagen filtrada.

        Excepciones
        -----------
        TypeError
            Si data no es un numpy array.

        ValueError
            Si la imagen no es 2-D ni 3-D.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un numpy array")

        if data.ndim not in (2, 3):
            raise ValueError("data debe ser 2-D o 3-D")

        if data.ndim == 2:
            return self._convolucion(data)

        canales = [
            self._convolucion(data[:, :, c])
            for c in range(data.shape[2])
        ]

        return np.stack(canales, axis=2)

    @property
    def tipo(self):
        """
        Retorna el tipo de filtro.

        Retorna
        -------
        str
            Nombre del filtro.
        """

        return self._tipo

    @property
    def kernel(self):
        """
        Retorna una copia del kernel del filtro.

        Retorna
        -------
        np.ndarray
            Kernel de convolución.
        """

        return self._kernel.copy()

    @property
    def tamaño(self):
        """
        Retorna el tamaño del kernel.

        Retorna
        -------
        int
            Tamaño del filtro.
        """

        return self._tamaño

    def __str__(self):
        """
        Retorna una representación textual del filtro.

        Retorna
        -------
        str
            Descripción resumida del filtro.
        """

        return f"Filtro(tipo='{self._tipo}', tamaño={self._tamaño})"

    def __repr__(self):
        """
        Retorna la representación oficial del objeto.

        Retorna
        -------
        str
            Representación textual del filtro.
        """

        return self.__str__()


class FiltroGaussiano(Filtro):
    """
    Filtro gaussiano para suavizado de imágenes.

    Este filtro aplica una convolución basada en una distribución
    gaussiana bidimensional, permitiendo reducir ruido y suavizar
    variaciones abruptas de intensidad.

    Atributos adicionales
    ---------------------
    _sigma : float
        Desviación estándar de la distribución gaussiana.
    """

    def __init__(self, tamaño=3, sigma=1.0):
        """
        Inicializa un filtro gaussiano.

        Parámetros
        ----------
        tamaño : int, opcional
            Tamaño del kernel.

        sigma : float, opcional
            Desviación estándar de la gaussiana.

        Excepciones
        -----------
        ValueError
            Si sigma no es positivo.
        """

        if sigma <= 0:
            raise ValueError("sigma debe ser positivo")

        self._sigma = float(sigma)

        super().__init__("gaussiano", tamaño)

    def _construir_kernel(self):
        """
        Construye el kernel gaussiano normalizado.

        Retorna
        -------
        np.ndarray
            Kernel gaussiano.
        """

        centro = self._tamaño // 2

        x = np.arange(-centro, centro + 1)
        y = np.arange(-centro, centro + 1)

        xx, yy = np.meshgrid(x, y)

        k = np.exp(-(xx**2 + yy**2) / (2 * self._sigma**2))

        k /= np.sum(k)

        return k


class FiltroMediana(Filtro):
    """
    Filtro de mediana para reducción de ruido impulsivo.

    Este filtro reemplaza cada píxel por la mediana de su vecindad,
    siendo especialmente útil para eliminar ruido tipo sal y pimienta.
    """

    def __init__(self, tamaño=3):
        """
        Inicializa un filtro de mediana.

        Parámetros
        ----------
        tamaño : int, opcional
            Tamaño de la ventana de filtrado.
        """

        super().__init__("mediana", tamaño)

    def _convolucion(self, canal):
        """
        Aplica filtrado por mediana sobre un canal.

        Parámetros
        ----------
        canal : np.ndarray
            Canal de entrada.

        Retorna
        -------
        np.ndarray
            Canal filtrado.
        """

        salida = median_filter(
            canal,
            size=self._tamaño,
            mode="reflect"
        )

        return salida.astype(canal.dtype)


class FiltroSuavizado(Filtro):
    """
    Filtro de suavizado promedio.

    Este filtro realiza un promedio espacial uniforme sobre
    la vecindad de cada píxel.
    """

    def __init__(self, tamaño=3):
        """
        Inicializa un filtro de suavizado.

        Parámetros
        ----------
        tamaño : int, opcional
            Tamaño del kernel promedio.
        """

        super().__init__("suavizado", tamaño)

    def _construir_kernel(self):
        """
        Construye un kernel promedio normalizado.

        Retorna
        -------
        np.ndarray
            Kernel uniforme de suavizado.
        """

        n = self._tamaño * self._tamaño

        return (
            np.ones(
                (self._tamaño, self._tamaño),
                dtype=np.float64
            ) / n
        )


class FiltroNitidez(Filtro):
    """
    Filtro de realce de nitidez.

    Este filtro incrementa el contraste local resaltando
    bordes y detalles finos de la imagen.
    """

    def __init__(self):
        """
        Inicializa un filtro de nitidez.

        El tamaño del kernel es fijo en 3x3.
        """

        super().__init__("nitidez", tamaño=3)

    def _construir_kernel(self):
        """
        Construye el kernel de nitidez.

        Retorna
        -------
        np.ndarray
            Kernel utilizado para realce de bordes.
        """

        return np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            dtype=np.float64
        )