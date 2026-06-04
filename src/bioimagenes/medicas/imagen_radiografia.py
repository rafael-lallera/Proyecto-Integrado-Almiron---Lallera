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
from matplotlib.patches import Circle
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
    # CLUSTERING DE MÚLTIPLES RADIOGRAFÍAS CON HOVER
    # =========================================================
    @staticmethod
    def graficar_clusters_imagenes(
        imagenes: list,
        k: int = 3,
        tamaño_thumbnail: int = 64
    ) -> None:
        """
        Agrupa una lista de ImagenRadiografia mediante k-means
        aplicado sobre features estadísticas de intensidad.

        Cada imagen se representa como un punto cuadrado en
        un espacio 2D (proyección PCA sobre 4 features).
        Al pasar el mouse sobre un punto se muestra la
        radiografía correspondiente como thumbnail inline,
        junto con sus coordenadas y cluster asignado.

        Features por imagen (vector de 4 valores):
        - Intensidad media
        - Desviación estándar de intensidades
        - Percentil 25 de intensidades
        - Percentil 75 de intensidades

        Parámetros
        ----------
        imagenes : list[ImagenRadiografia]
            Lista de objetos ImagenRadiografia a agrupar.
            Mínimo 2 imágenes, debe tener al menos k imágenes.
        k : int
            Número de clusters (default=3).
        tamaño_thumbnail : int
            Tamaño en píxeles del thumbnail en el hover
            (default=64).
        """
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from PIL import Image as PILImage

        # --------------------------------------------------
        # VALIDACIONES
        # --------------------------------------------------
        if not isinstance(imagenes, list):
            raise TypeError("imagenes debe ser una lista")
        if len(imagenes) < 2:
            raise ValueError(
                "Se necesitan al menos 2 imágenes"
            )
        if not isinstance(k, int):
            raise TypeError("k debe ser entero")
        if k < 2:
            raise ValueError("k debe ser >= 2")
        if k > len(imagenes):
            raise ValueError(
                f"k={k} no puede ser mayor que "
                f"la cantidad de imágenes ({len(imagenes)})"
            )
        if not isinstance(tamaño_thumbnail, int):
            raise TypeError(
                "tamaño_thumbnail debe ser entero"
            )
        if tamaño_thumbnail < 16:
            raise ValueError(
                "tamaño_thumbnail debe ser >= 16"
            )

        # --------------------------------------------------
        # 1. EXTRACCIÓN DE FEATURES Y THUMBNAILS
        # --------------------------------------------------
        features = []
        thumbnails = []

        for img in imagenes:
            d = img._data.flatten()
            vec = np.array([
                float(np.mean(d)),
                float(np.std(d)),
                float(np.percentile(d, 25)),
                float(np.percentile(d, 75)),
            ])
            features.append(vec)

            # Thumbnail redimensionado para hover
            pil = PILImage.fromarray(
                img._data.astype(np.uint8)
            ).resize(
                (tamaño_thumbnail * 2, tamaño_thumbnail * 2),
                PILImage.LANCZOS
            )
            thumbnails.append(np.array(pil))

        features = np.array(features, dtype=np.float64)

        # --------------------------------------------------
        # 2. NORMALIZACIÓN DE FEATURES
        # --------------------------------------------------
        media_f = features.mean(axis=0)
        std_f = features.std(axis=0)
        std_f[std_f == 0] = 1.0
        features_norm = (features - media_f) / std_f

        # --------------------------------------------------
        # 3. K-MEANS
        # --------------------------------------------------
        centroides_feat, etiquetas = kmeans2(
            features_norm,
            k,
            minit="points",
            iter=50
        )

        # --------------------------------------------------
        # 4. PROYECCIÓN PCA A 2D
        # --------------------------------------------------
        datos_centrados = (
            features_norm - features_norm.mean(axis=0)
        )
        covarianza = np.cov(datos_centrados.T)

        if covarianza.ndim < 2:
            covarianza = np.atleast_2d(covarianza)

        valores, vectores = np.linalg.eigh(covarianza)
        idx_orden = np.argsort(valores)[::-1]
        componentes = vectores[:, idx_orden[:2]]

        puntos_2d = datos_centrados @ componentes
        centroides_2d = (
            (centroides_feat - features_norm.mean(axis=0))
            @ componentes
        )

        # --------------------------------------------------
        # 5. JITTER: separar puntos que se solapan
        #    (útil cuando hay pocas imágenes)
        # --------------------------------------------------
        rng = np.random.default_rng(42)
        rango = puntos_2d.max(axis=0) - puntos_2d.min(axis=0)
        escala_jitter = np.where(rango > 0, rango, 1.0) * 0.08
        puntos_2d = (
            puntos_2d
            + rng.uniform(
                -escala_jitter,
                escala_jitter,
                size=puntos_2d.shape
            )
        )

        # --------------------------------------------------
        # 6. PALETA DE COLORES
        # --------------------------------------------------
        paleta = [
            "#1f77b4",  # azul
            "#d62728",  # rojo
            "#2ca02c",  # verde
            "#9467bd",  # violeta
            "#ff7f0e",  # naranja
            "#8c564b",  # marrón
            "#e377c2",  # rosa
        ]
        colores_puntos = [
            paleta[int(etiquetas[i]) % len(paleta)]
            for i in range(len(imagenes))
        ]

        # --------------------------------------------------
        # 7. FIGURA
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 8))
        plt.subplots_adjust(right=0.78)

        # Scatter principal (marcador cuadrado)
        sc = ax.scatter(
            puntos_2d[:, 0],
            puntos_2d[:, 1],
            c=colores_puntos,
            s=140,
            marker="s",
            alpha=0.88,
            zorder=3,
            edgecolors="white",
            linewidths=0.8
        )

        # Centroides con X negra
        for i in range(k):
            ax.scatter(
                centroides_2d[i, 0],
                centroides_2d[i, 1],
                marker="X",
                s=320,
                c="black",
                zorder=5,
                edgecolors="white",
                linewidths=1.2
            )

        # Leyenda de clusters
        handles_leyenda = [
            plt.Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=paleta[i % len(paleta)],
                markersize=11,
                label=f"Cluster {i}"
            )
            for i in range(k)
        ]
        ax.legend(
            handles=handles_leyenda,
            loc="upper right",
            bbox_to_anchor=(1.22, 1.0),
            title="Clusters",
            framealpha=0.9
        )

        ax.set_title(
            "Image Clusters",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # --------------------------------------------------
        # 8. HOVER CON OffsetImage + AnnotationBbox
        #    Este es el approach correcto para thumbnails
        #    inline en matplotlib sin abrir ventanas nuevas
        # --------------------------------------------------

        # AnnotationBbox invisible al inicio
        imagen_hover = OffsetImage(
            thumbnails[0],
            zoom=1.0,
            cmap="gray"
        )
        imagen_hover.image.axes = ax

        ab = AnnotationBbox(
            imagen_hover,
            (0, 0),
            xybox=(60, -60),
            xycoords="data",
            boxcoords="offset points",
            pad=0.4,
            bboxprops=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.95
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1.0
            )
        )
        ab.set_visible(False)
        ax.add_artist(ab)

        # Texto con info del punto
        info_box = ax.text(
            0, 0,
            "",
            fontsize=8.5,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="gray",
                alpha=0.92,
                linewidth=0.8
            ),
            zorder=10,
            visible=False
        )

        ultimo_idx = [-1]

        def on_move(event):
            if event.inaxes != ax:
                if ab.get_visible():
                    ab.set_visible(False)
                    info_box.set_visible(False)
                    fig.canvas.draw_idle()
                return

            # Distancia en coordenadas de pantalla
            if len(puntos_2d) == 0:
                return

            try:
                xy_pantalla = ax.transData.transform(puntos_2d)
            except Exception:
                return

            dx = xy_pantalla[:, 0] - event.x
            dy = xy_pantalla[:, 1] - event.y
            dist = np.hypot(dx, dy)
            idx = int(np.argmin(dist))

            UMBRAL_PX = 22

            if dist[idx] > UMBRAL_PX:
                if ab.get_visible():
                    ab.set_visible(False)
                    info_box.set_visible(False)
                    fig.canvas.draw_idle()
                ultimo_idx[0] = -1
                return

            if idx == ultimo_idx[0]:
                return

            ultimo_idx[0] = idx

            # Actualizar imagen del hover
            imagen_hover.set_data(thumbnails[idx])

            # Posición del annotation (punto de anclaje)
            px = puntos_2d[idx, 0]
            py = puntos_2d[idx, 1]
            ab.xy = (px, py)

            # Offset dinámico: si el punto está en la mitad
            # derecha del eje, mostrar el tooltip a la izquierda
            xlim = ax.get_xlim()
            mitad_x = (xlim[0] + xlim[1]) / 2.0
            offset_x = -80 if px > mitad_x else 80
            ab.xybox = (offset_x, -70)

            ab.set_visible(True)

            # Texto informativo
            cluster_n = int(etiquetas[idx])
            info_box.set_text(
                f"X: {px:.2f}\n"
                f"Y: {py:.2f}\n"
                f"Cluster: {cluster_n}"
            )

            # Posicionar el texto cerca del punto
            ylim = ax.get_ylim()
            rango_y = ylim[1] - ylim[0]
            info_box.set_position((px, py + rango_y * 0.04))
            info_box.set_visible(True)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect(
            "motion_notify_event",
            on_move
        )

        plt.tight_layout()
        plt.show()