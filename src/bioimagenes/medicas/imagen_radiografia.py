"""
Subclase de Imagen especializada en radiografías (rayos X).

La clase trabaja exclusivamente con imágenes 2-D en escala
de grises y mantiene internamente los datos como float64
en el rango [0, 255].

Se incorporan operaciones típicas de procesamiento
radiográfico:

- Ajuste de visualización.
- Normalización.
- Mejora de contraste.
- Inversión de intensidad.
- Ecualización de histograma.
- Detección de bordes Sobel.
- Selección de ROI.
- Clustering k-means.

Dependencias
------------
- numpy
- matplotlib.pyplot
- scipy.signal.convolve2d
- scipy.cluster.vq.kmeans2
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy.cluster.vq import kmeans2

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


class ImagenRadiografia(Imagen):

    _RANGO_INTENSIDAD = (0.0, 255.0)

    # =========================================================
    # CONSTRUCTOR
    # =========================================================
    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        tipo_estudio: str = "no especificado",
        condiciones_adquisicion: str = "no especificadas",
    ):

        # -----------------------------------------------------
        # VALIDACIONES
        # -----------------------------------------------------
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un numpy array")

        if data.ndim != 2:
            raise ValueError(
                "ImagenRadiografia requiere una imagen 2-D"
            )

        if not isinstance(tipo_estudio, str):
            raise TypeError("tipo_estudio debe ser string")

        if not isinstance(condiciones_adquisicion, str):
            raise TypeError(
                "condiciones_adquisicion debe ser string"
            )

        # -----------------------------------------------------
        # CONVERSIÓN INTERNA
        # -----------------------------------------------------
        data = data.astype(np.float64)

        # -----------------------------------------------------
        # COMPATIBILIDAD CON PNG NORMALIZADOS
        # -----------------------------------------------------
        #
        # matplotlib.pyplot.imread() suele cargar PNG
        # como float32 en rango [0,1].
        #
        # La clase ImagenRadiografia trabaja internamente
        # en rango [0,255], por lo que se convierte
        # automáticamente si corresponde.
        #
        max_val = float(np.max(data))

        if max_val <= 1.0:
            data = data * 255.0

        # -----------------------------------------------------
        # CLIP DE SEGURIDAD
        # -----------------------------------------------------
        data = np.clip(
            data,
            self._RANGO_INTENSIDAD[0],
            self._RANGO_INTENSIDAD[1]
        )

        super().__init__(data, info)

        self._tipo_estudio = tipo_estudio
        self._condiciones_adquisicion = (
            condiciones_adquisicion
        )

        self._info["historial"].modificar_historial(
            f"Radiografía creada — estudio: {tipo_estudio}"
        )

    # =========================================================
    # PROPIEDADES
    # =========================================================
    @property
    def tipo_estudio(self) -> str:
        return self._tipo_estudio

    @property
    def condiciones_adquisicion(self) -> str:
        return self._condiciones_adquisicion

    # =========================================================
    # NORMALIZAR
    # =========================================================
    def normalizar(self) -> None:
        """
        Normaliza la radiografía al rango [0,255].
        """

        min_val = float(np.min(self._data))
        max_val = float(np.max(self._data))

        if max_val - min_val == 0:
            raise ValueError(
                "Imagen constante: no se puede normalizar"
            )

        self._data = (
            (self._data - min_val)
            / (max_val - min_val)
            * 255.0
        ).astype(np.float64)

        self._info["historial"].modificar_historial(
            "Normalización radiográfica aplicada"
        )

    # =========================================================
    # AJUSTAR VISUALIZACIÓN
    # =========================================================
    def ajustar_visualizacion(
        self,
        vmin: float = 0.0,
        vmax: float = 255.0
    ) -> None:

        if not isinstance(vmin, (int, float)):
            raise TypeError("vmin debe ser numérico")

        if not isinstance(vmax, (int, float)):
            raise TypeError("vmax debe ser numérico")

        if vmin >= vmax:
            raise ValueError(
                "vmin debe ser menor que vmax"
            )

        self._data = np.clip(
            self._data,
            vmin,
            vmax
        )

        self._data = (
            (self._data - vmin)
            / (vmax - vmin)
            * 255.0
        )

        self._info["historial"].modificar_historial(
            f"Visualización ajustada: "
            f"vmin={vmin}, vmax={vmax}"
        )

    # =========================================================
    # MEJORAR CONTRASTE
    # =========================================================
    def mejorar_contraste(self) -> None:
        """
        Expande el histograma al rango completo [0,255].
        """

        min_val = float(np.min(self._data))
        max_val = float(np.max(self._data))

        if max_val == min_val:
            raise ValueError(
                "Imagen constante: no se puede mejorar"
            )

        self._data = (
            (self._data - min_val)
            / (max_val - min_val)
            * 255.0
        )

        self._info["historial"].modificar_historial(
            "Contraste mejorado"
        )

    # =========================================================
    # INVERTIR INTENSIDAD
    # =========================================================
    def invertir_intensidad(self) -> None:
        """
        Genera un negativo radiográfico.
        """

        self._data = 255.0 - self._data

        self._info["historial"].modificar_historial(
            "Intensidades invertidas"
        )

    # =========================================================
    # ECUALIZAR HISTOGRAMA
    # =========================================================
    def ecualizar_histograma(self) -> None:
        """
        Ecualización global mediante CDF.
        """

        data_uint = self._data.astype(np.uint8)

        histograma, _ = np.histogram(
            data_uint.flatten(),
            bins=256,
            range=(0, 256)
        )

        if np.count_nonzero(histograma) <= 1:
            raise ValueError(
                "Imagen constante: no se puede ecualizar"
            )

        cdf = histograma.cumsum()

        cdf_min = cdf[cdf > 0].min()

        denominador = self._data.size - cdf_min

        if denominador == 0:
            raise ValueError(
                "No se puede ecualizar la imagen"
            )

        cdf_norm = (
            (cdf - cdf_min)
            / denominador
            * 255.0
        )

        cdf_norm = np.clip(cdf_norm, 0, 255)

        self._data = (
            cdf_norm[data_uint]
        ).astype(np.float64)

        self._info["historial"].modificar_historial(
            "Histograma ecualizado"
        )

    # =========================================================
    # DETECTAR BORDES
    # =========================================================
    def detectar_bordes(self) -> np.ndarray:
        """
        Detección de bordes mediante Sobel.
        """

        kx = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=np.float64
        )

        ky = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=np.float64
        )

        gx = convolve2d(
            self._data,
            kx,
            mode="same",
            boundary="symm"
        )

        gy = convolve2d(
            self._data,
            ky,
            mode="same",
            boundary="symm"
        )

        magnitud = np.hypot(gx, gy)

        magnitud = np.clip(
            magnitud,
            0.0,
            255.0
        )

        self._info["historial"].modificar_historial(
            "Detección de bordes Sobel"
        )

        return magnitud.astype(np.float64)

    # =========================================================
    # ROI
    # =========================================================
    def seleccionar_roi(
        self,
        x: int,
        y: int,
        ancho: int,
        alto: int
    ) -> "ImagenRadiografia":

        if not all(
            isinstance(v, int)
            for v in (x, y, ancho, alto)
        ):
            raise TypeError(
                "x, y, ancho y alto deben ser enteros"
            )

        if ancho <= 0 or alto <= 0:
            raise ValueError(
                "ancho y alto deben ser positivos"
            )

        filas, columnas = self._data.shape

        if x < 0 or y < 0:
            raise ValueError(
                "x e y deben ser positivos"
            )

        if x + alto > filas or y + ancho > columnas:
            raise ValueError(
                "La ROI excede la imagen"
            )

        roi = self._data[
            x:x + alto,
            y:y + ancho
        ].copy()

        nueva_info = Info(
            dimensiones=roi.shape,
            brillo=self._info["brillo"],
            ruta_origen=self._info["ruta_origen"]
        )

        nueva_info["historial"].modificar_historial(
            f"ROI seleccionada: "
            f"x={x}, y={y}, "
            f"ancho={ancho}, alto={alto}"
        )

        return ImagenRadiografia(
            roi,
            nueva_info,
            tipo_estudio=self._tipo_estudio,
            condiciones_adquisicion=(
                self._condiciones_adquisicion
            )
        )

    # =========================================================
    # REGISTRAR TRANSFORMACIÓN
    # =========================================================
    def registrar_transformacion(
        self,
        descripcion: str
    ) -> None:

        if not isinstance(descripcion, str):
            raise TypeError(
                "descripcion debe ser string"
            )

        if not descripcion.strip():
            raise ValueError(
                "descripcion vacía"
            )

        self._info["historial"].modificar_historial(
            descripcion
        )

    # =========================================================
    # VISUALIZAR
    # =========================================================
    def visualizar(
        self,
        slice_index=None
    ) -> None:
        """
        Visualiza la radiografía.
        """

        plt.figure(figsize=(6, 6))

        plt.imshow(
            self._data,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title(
            f"Radiografía — {self._tipo_estudio}\n"
            f"Condiciones: "
            f"{self._condiciones_adquisicion}"
        )

        plt.axis("off")

        plt.colorbar(label="Intensidad")

        plt.tight_layout()

        plt.show()

    # =========================================================
    # CLUSTERING
    # =========================================================
    def graficar_clusters(
        self,
        k: int = 3
    ) -> None:

        if not isinstance(k, int):
            raise TypeError("k debe ser entero")

        if k < 2:
            raise ValueError(
                "k debe ser >= 2"
            )

        datos = self._data.reshape(-1, 1)

        centroides, etiquetas = kmeans2(
            datos.astype(np.float64),
            k,
            minit="points"
        )

        clusters = etiquetas.reshape(
            self._data.shape
        )

        plt.figure(figsize=(6, 6))

        plt.imshow(
            clusters,
            cmap="gray"
        )

        plt.title(
            f"Clusters de intensidad (k={k})"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

        self._info["historial"].modificar_historial(
            f"Clustering k-means aplicado (k={k})"
        )

    # =========================================================
    # REPRESENTACIÓN
    # =========================================================
    def __str__(self) -> str:

        return (
            f"ImagenRadiografia("
            f"shape={self._data.shape}, "
            f"estudio='{self._tipo_estudio}', "
            f"condiciones="
            f"'{self._condiciones_adquisicion}')"
        )