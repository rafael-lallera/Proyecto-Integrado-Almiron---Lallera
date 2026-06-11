"""
bioimagenes/medicas/imagen_tomografia.py

Subclase de Imagen especializada en tomografías (CT).

La clase trabaja con volúmenes 3-D almacenados en formato
NIfTI (.nii / .nii.gz) o como numpy arrays precargados.
Internamente los datos se representan como float64.

Las intensidades siguen la escala Hounsfield (HU):
  - Aire:          ~ -1000 HU
  - Pulmón:        -900 a -500 HU
  - Grasa:         -100 a  -50 HU
  - Agua/tejido:      0 a   80 HU
  - Músculo:         40 a   80 HU
  - Hueso esponjoso: 100 a  400 HU
  - Hueso cortical:  400 a 1900 HU

Operaciones soportadas:
  - Carga desde archivo NIfTI.
  - Acceso a cortes individuales por eje (axial/coronal/sagital).
  - Visualización de cortes en escala de grises.
  - Ajuste de ventana (window center / window width).
  - Presets de tejido (pulmón, hueso, tejido blando, cerebro).
  - Segmentación por colores según tipo de tejido (visualizar_corte).
  - Reconstrucción 3-D simplificada (proyección MIP).
  - Normalización de intensidades.
  - Registro de transformaciones.

Dependencias
------------
- numpy
- matplotlib.pyplot
- nibabel  (lectura NIfTI)
- bioimagenes.core.imagen
- bioimagenes.core.info

Autor
-----
Proyecto Bioimágenes
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


# =============================================================
# TABLA DE TEJIDOS
# =============================================================
#
# Cada entrada define:
#   nombre  : etiqueta visible
#   hu_min  : límite inferior en HU
#   hu_max  : límite superior en HU
#   color   : color RGB normalizado [0,1] para la máscara
#
_TEJIDOS = [
    {
        "nombre": "Aire",
        "hu_min": -2000,
        "hu_max": -900,
        "color": (0.0,  0.0,  0.0),   # negro
    },
    {
        "nombre": "Pulmón",
        "hu_min": -900,
        "hu_max": -500,
        "color": (0.2,  0.4,  1.0),   # azul
    },
    {
        "nombre": "Grasa",
        "hu_min": -500,
        "hu_max": -50,
        "color": (1.0,  0.85, 0.0),   # amarillo
    },
    {
        "nombre": "Tejido blando",
        "hu_min": -50,
        "hu_max": 100,
        "color": (0.2,  0.75, 0.2),   # verde
    },
    {
        "nombre": "Músculo",
        "hu_min": 100,
        "hu_max": 300,
        "color": (0.85, 0.1,  0.1),   # rojo
    },
    {
        "nombre": "Hueso esponjoso",
        "hu_min": 300,
        "hu_max": 700,
        "color": (1.0,  0.5,  0.9),   # rosa
    },
    {
        "nombre": "Hueso cortical",
        "hu_min": 700,
        "hu_max": 3000,
        "color": (1.0,  1.0,  1.0),   # blanco
    },
]

# Presets de ventana (window_center, window_width) en HU
_PRESETS_TEJIDO: dict[str, tuple[float, float]] = {
    "pulmon":        (-600.0,  1500.0),
    "hueso":         (400.0,   1800.0),
    "tejido_blando": (50.0,    400.0),
    "cerebro":       (40.0,    80.0),
    "abdomen":       (60.0,    400.0),
    "mediastino":    (50.0,    350.0),
}


class ImagenTomografia(Imagen):
    """
    Imagen tomográfica (CT) con soporte para volúmenes 3-D.

    Parámetros
    ----------
    data : np.ndarray
        Volumen 3-D (slices, filas, columnas) o corte 2-D,
        con valores en unidades Hounsfield (HU).
    info : Info
        Metadatos del volumen.
    voxel_spacing : tuple[float, float, float], opcional
        Tamaño físico de cada voxel en mm
        (slice_thickness, row_spacing, col_spacing).
        Default: (1.0, 1.0, 1.0).
    origen : tuple[float, float, float], opcional
        Coordenadas físicas del voxel (0,0,0). Default: (0,0,0).
    escala_intensidad : str, opcional
        Descripción de la escala de intensidades.
        Default: "Hounsfield".
    window_center : float, opcional
        Centro de la ventana de visualización. Default: 40.0.
    window_width : float, opcional
        Ancho de la ventana de visualización. Default: 400.0.
    """

    # =========================================================
    # CONSTRUCTOR
    # =========================================================
    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        voxel_spacing: tuple = (1.0, 1.0, 1.0),
        origen: tuple = (0.0, 0.0, 0.0),
        escala_intensidad: str = "Hounsfield",
        window_center: float = 40.0,
        window_width: float = 400.0,
    ):
        # -----------------------------------------------------
        # VALIDACIONES PROPIAS
        # -----------------------------------------------------
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un numpy array")
        if data.ndim not in (2, 3):
            raise ValueError(
                "ImagenTomografia requiere datos 2-D o 3-D"
            )
        if not isinstance(voxel_spacing, (tuple, list)):
            raise TypeError("voxel_spacing debe ser tupla")
        if len(voxel_spacing) != 3:
            raise ValueError(
                "voxel_spacing debe tener 3 componentes"
            )
        if not isinstance(escala_intensidad, str):
            raise TypeError(
                "escala_intensidad debe ser string"
            )
        if not isinstance(window_center, (int, float)):
            raise TypeError(
                "window_center debe ser numérico"
            )
        if not isinstance(window_width, (int, float)):
            raise TypeError(
                "window_width debe ser numérico"
            )
        if window_width <= 0:
            raise ValueError(
                "window_width debe ser positivo"
            )

        # -----------------------------------------------------
        # CONVERSIÓN INTERNA
        # -----------------------------------------------------
        data = data.astype(np.float64)

        # -----------------------------------------------------
        # LLAMADA AL PADRE
        # -----------------------------------------------------
        super().__init__(data, info)

        # -----------------------------------------------------
        # ATRIBUTOS PROPIOS
        # -----------------------------------------------------
        self._voxel_spacing = tuple(
            float(v) for v in voxel_spacing
        )
        self._origen = tuple(float(v) for v in origen)
        self._escala_intensidad = escala_intensidad
        self._window_center = float(window_center)
        self._window_width = float(window_width)
        self._presets_tejido = dict(_PRESETS_TEJIDO)

        self._info["historial"].modificar_historial(
            f"Tomografía creada — escala: {escala_intensidad}, "
            f"shape: {data.shape}"
        )

    # =========================================================
    # PROPIEDADES
    # =========================================================
    @property
    def voxel_spacing(self) -> tuple:
        return self._voxel_spacing

    @property
    def origen(self) -> tuple:
        return self._origen

    @property
    def escala_intensidad(self) -> str:
        return self._escala_intensidad

    @property
    def window_center(self) -> float:
        return self._window_center

    @property
    def window_width(self) -> float:
        return self._window_width

    @property
    def presets_tejido(self) -> dict:
        return dict(self._presets_tejido)

    @property
    def n_slices(self) -> int:
        """Cantidad de cortes en el eje axial (eje 0)."""
        if self._data.ndim == 3:
            return self._data.shape[0]
        return 1

    # =========================================================
    # CARGA DESDE NIFTI
    # =========================================================
    @classmethod
    def desde_nifti(
        cls,
        ruta: str,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> "ImagenTomografia":
        """
        Carga un volumen desde un archivo NIfTI (.nii / .nii.gz).

        El volumen se reorienta a (slices, filas, columnas),
        es decir eje axial primero.

        Parámetros
        ----------
        ruta : str
            Ruta al archivo .nii o .nii.gz.
        window_center : float
            Centro de ventana inicial. Default: 40.
        window_width : float
            Ancho de ventana inicial. Default: 400.

        Retorna
        -------
        ImagenTomografia
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel no está instalado. "
                "Ejecutá: pip install nibabel"
            )

        nii = nib.load(ruta)
        volumen = np.array(nii.get_fdata(), dtype=np.float64)

        # NIfTI almacena (x, y, z) → reordenar a (z, y, x)
        # para que el eje 0 sea el eje de los slices axiales
        if volumen.ndim == 3:
            volumen = np.transpose(volumen, (2, 1, 0))

        # Espaciado de voxel desde header
        header = nii.header
        try:
            pixdim = header.get_zooms()
            if len(pixdim) >= 3:
                spacing = (
                    float(pixdim[2]),
                    float(pixdim[1]),
                    float(pixdim[0]),
                )
            else:
                spacing = (1.0, 1.0, 1.0)
        except Exception:
            spacing = (1.0, 1.0, 1.0)

        info = Info(
            dimensiones=volumen.shape,
            brillo=1.0,
            ruta_origen=ruta
        )

        return cls(
            data=volumen,
            info=info,
            voxel_spacing=spacing,
            window_center=window_center,
            window_width=window_width,
        )

    # =========================================================
    # OBTENER CORTE
    # =========================================================
    def obtener_corte(
        self,
        indice: int,
        eje: str = "axial"
    ) -> np.ndarray:
        """
        Retorna un corte 2-D del volumen.

        Parámetros
        ----------
        indice : int
            Índice del corte.
        eje : str
            "axial" (eje 0), "coronal" (eje 1)
            o "sagital" (eje 2).

        Retorna
        -------
        np.ndarray
            Corte 2-D en float64.
        """
        if self._data.ndim == 2:
            return self._data.copy()

        ejes_validos = ("axial", "coronal", "sagital")
        if eje not in ejes_validos:
            raise ValueError(
                f"eje debe ser uno de {ejes_validos}"
            )

        eje_idx = {"axial": 0, "coronal": 1, "sagital": 2}[eje]

        if not isinstance(indice, int):
            raise TypeError("indice debe ser entero")

        limite = self._data.shape[eje_idx]
        if indice < 0 or indice >= limite:
            raise ValueError(
                f"indice {indice} fuera de rango "
                f"[0, {limite - 1}] para eje '{eje}'"
            )

        if eje == "axial":
            return self._data[indice, :, :].copy()
        elif eje == "coronal":
            return self._data[:, indice, :].copy()
        else:
            return self._data[:, :, indice].copy()

    # =========================================================
    # APLICAR VENTANA
    # =========================================================
    def _aplicar_ventana(
        self,
        corte: np.ndarray,
        center: float = None,
        width: float = None,
    ) -> np.ndarray:
        """
        Aplica windowing sobre un corte y retorna
        valores en [0, 255].

        Parámetros
        ----------
        corte : np.ndarray
            Corte 2-D en HU.
        center : float, opcional
            Window center. Default: self._window_center.
        width : float, opcional
            Window width. Default: self._window_width.

        Retorna
        -------
        np.ndarray
            Corte normalizado a [0, 255] float64.
        """
        if center is None:
            center = self._window_center
        if width is None:
            width = self._window_width

        hu_min = center - width / 2.0
        hu_max = center + width / 2.0

        corte_w = np.clip(corte, hu_min, hu_max)
        corte_w = (
            (corte_w - hu_min) / (hu_max - hu_min) * 255.0
        )
        return corte_w.astype(np.float64)

    # =========================================================
    # MOSTRAR SLICE
    # =========================================================
    def mostrar_slice(
        self,
        indice: int,
        eje: str = "axial",
        center: float = None,
        width: float = None,
    ) -> None:
        """
        Visualiza un corte individual en escala de grises
        aplicando windowing.

        Parámetros
        ----------
        indice : int
            Índice del corte.
        eje : str
            "axial", "coronal" o "sagital".
        center : float, opcional
            Window center. Default: self._window_center.
        width : float, opcional
            Window width. Default: self._window_width.
        """
        corte = self.obtener_corte(indice, eje)
        corte_w = self._aplicar_ventana(corte, center, width)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            corte_w,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal"
        )
        plt.colorbar(im, ax=ax, label="Intensidad (ventana)")
        ax.set_title(
            f"Corte {eje} #{indice} — "
            f"WC={self._window_center:.0f} "
            f"WW={self._window_width:.0f}"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        self._info["historial"].modificar_historial(
            f"Mostrado corte {eje} #{indice}"
        )

    # =========================================================
    # VISUALIZAR CORTE (SEGMENTACIÓN POR TEJIDOS)
    # =========================================================
    def visualizar_corte(
        self,
        indice: int,
        eje: str = "axial",
        mostrar_original: bool = True,
    ) -> np.ndarray:
        """
        Segmenta un corte por tipo de tejido usando rangos HU
        y lo visualiza como imagen RGB coloreada.

        Genera una imagen RGB donde cada tejido tiene un color
        distinto según la tabla de Hounsfield, tal como se
        muestra en la Figura 1 de referencia.

        Parámetros
        ----------
        indice : int
            Índice del corte a segmentar.
        eje : str
            "axial", "coronal" o "sagital".
        mostrar_original : bool
            Si True, muestra el corte original en grises
            junto a la segmentación. Default: True.

        Retorna
        -------
        np.ndarray
            Matriz RGB (filas, columnas, 3) float64 en [0,1]
            con los tejidos coloreados.
        """
        corte = self.obtener_corte(indice, eje)
        filas, columnas = corte.shape

        # --------------------------------------------------
        # CONSTRUIR IMAGEN RGB
        # --------------------------------------------------
        rgb = np.zeros((filas, columnas, 3), dtype=np.float64)

        for tejido in _TEJIDOS:
            mascara = (
                (corte >= tejido["hu_min"])
                & (corte < tejido["hu_max"])
            )
            r, g, b = tejido["color"]
            rgb[mascara, 0] = r
            rgb[mascara, 1] = g
            rgb[mascara, 2] = b

        # --------------------------------------------------
        # VISUALIZACIÓN
        # --------------------------------------------------
        if mostrar_original:
            fig, axes = plt.subplots(
                1, 2,
                figsize=(12, 6)
            )
            corte_w = self._aplicar_ventana(corte)
            axes[0].imshow(
                corte_w,
                cmap="gray",
                vmin=0,
                vmax=255
            )
            axes[0].set_title(
                f"Original — corte {eje} #{indice}"
            )
            axes[0].axis("off")

            axes[1].imshow(rgb)
            axes[1].set_title(
                f"Segmentación por tejidos — corte {eje} #{indice}"
            )
            axes[1].axis("off")

            # Leyenda de tejidos
            parches = [
                mpatches.Patch(
                    color=t["color"],
                    label=(
                        f"{t['nombre']} "
                        f"[{t['hu_min']}, {t['hu_max']}] HU"
                    )
                )
                for t in _TEJIDOS
            ]
            axes[1].legend(
                handles=parches,
                loc="lower right",
                fontsize=7,
                framealpha=0.85
            )
            ax_ref = axes[1]

        else:
            fig, ax_ref = plt.subplots(figsize=(7, 7))
            ax_ref.imshow(rgb)
            ax_ref.set_title(
                f"Segmentación por tejidos — "
                f"corte {eje} #{indice}"
            )
            ax_ref.axis("off")

            parches = [
                mpatches.Patch(
                    color=t["color"],
                    label=(
                        f"{t['nombre']} "
                        f"[{t['hu_min']}, {t['hu_max']}] HU"
                    )
                )
                for t in _TEJIDOS
            ]
            ax_ref.legend(
                handles=parches,
                loc="lower right",
                fontsize=7,
                framealpha=0.85
            )

        plt.tight_layout()
        plt.show()

        self._info["historial"].modificar_historial(
            f"Segmentación por tejidos — corte {eje} #{indice}"
        )

        return rgb

    # =========================================================
    # AJUSTAR VENTANA
    # =========================================================
    def ajustar_ventana(
        self,
        center: float,
        width: float,
    ) -> None:
        """
        Actualiza los parámetros de windowing.

        Parámetros
        ----------
        center : float
            Nuevo window center en HU.
        width : float
            Nuevo window width en HU (debe ser > 0).
        """
        if not isinstance(center, (int, float)):
            raise TypeError("center debe ser numérico")
        if not isinstance(width, (int, float)):
            raise TypeError("width debe ser numérico")
        if width <= 0:
            raise ValueError("width debe ser positivo")

        self._window_center = float(center)
        self._window_width = float(width)

        self._info["historial"].modificar_historial(
            f"Ventana ajustada: WC={center}, WW={width}"
        )

    # =========================================================
    # APLICAR PRESET DE TEJIDO
    # =========================================================
    def aplicar_preset(
        self,
        tejido: str
    ) -> None:
        """
        Aplica un preset de ventana predefinido.

        Presets disponibles:
          "pulmon", "hueso", "tejido_blando",
          "cerebro", "abdomen", "mediastino"

        Parámetros
        ----------
        tejido : str
            Nombre del preset.
        """
        if not isinstance(tejido, str):
            raise TypeError("tejido debe ser string")

        tejido_lower = tejido.lower()
        if tejido_lower not in self._presets_tejido:
            disponibles = list(self._presets_tejido.keys())
            raise ValueError(
                f"Preset '{tejido}' no existe. "
                f"Disponibles: {disponibles}"
            )

        center, width = self._presets_tejido[tejido_lower]
        self.ajustar_ventana(center, width)

        self._info["historial"].modificar_historial(
            f"Preset aplicado: {tejido_lower} "
            f"(WC={center}, WW={width})"
        )

    # =========================================================
    # NORMALIZAR INTENSIDADES
    # =========================================================
    def normalizar_intensidades(
        self,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
    ) -> None:
        """
        Normaliza el volumen al rango [0, 1] recortando
        primero al rango HU especificado.

        Parámetros
        ----------
        hu_min : float
            Límite inferior de HU a considerar. Default: -1000.
        hu_max : float
            Límite superior de HU a considerar. Default: 1000.
        """
        if not isinstance(hu_min, (int, float)):
            raise TypeError("hu_min debe ser numérico")
        if not isinstance(hu_max, (int, float)):
            raise TypeError("hu_max debe ser numérico")
        if hu_min >= hu_max:
            raise ValueError(
                "hu_min debe ser menor que hu_max"
            )

        self._data = np.clip(
            self._data, hu_min, hu_max
        )
        self._data = (
            (self._data - hu_min)
            / (hu_max - hu_min)
        ).astype(np.float64)

        self._info["historial"].modificar_historial(
            f"Intensidades normalizadas: "
            f"HU [{hu_min}, {hu_max}] → [0, 1]"
        )

    # =========================================================
    # RECONSTRUCCIÓN 3D (MIP — proyección de máxima intensidad)
    # =========================================================
    def reconstruir_3d(
        self,
        eje: int = 0,
        center: float = None,
        width: float = None,
    ) -> None:
        """
        Genera una proyección de máxima intensidad (MIP)
        a lo largo del eje indicado como representación 3-D
        simplificada.

        Parámetros
        ----------
        eje : int
            Eje de proyección: 0=axial, 1=coronal, 2=sagital.
        center : float, opcional
            Window center para la visualización.
        width : float, opcional
            Window width para la visualización.
        """
        if self._data.ndim != 3:
            raise ValueError(
                "reconstruir_3d requiere volumen 3-D"
            )
        if eje not in (0, 1, 2):
            raise ValueError("eje debe ser 0, 1 o 2")

        mip = np.max(self._data, axis=eje)
        mip_w = self._aplicar_ventana(mip, center, width)

        nombres_eje = {
            0: "axial (MIP)",
            1: "coronal (MIP)",
            2: "sagital (MIP)"
        }

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(
            mip_w,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal"
        )
        plt.colorbar(im, ax=ax, label="Intensidad")
        ax.set_title(
            f"Reconstrucción 3-D — proyección {nombres_eje[eje]}"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        self._info["historial"].modificar_historial(
            f"Reconstrucción 3-D MIP — eje {eje}"
        )

    # =========================================================
    # VISUALIZAR (override de Imagen)
    # =========================================================
    def visualizar(
        self,
        slice_index: int = None,
        modo: str = None,
    ) -> None:
        """
        Visualiza la tomografía.

        - Si es 2-D, muestra el corte directamente.
        - Si es 3-D, muestra el corte central del eje axial
          (o el indicado por slice_index).

        Parámetros
        ----------
        slice_index : int, opcional
            Índice del corte axial a mostrar.
            Default: corte central.
        modo : str, opcional
            No utilizado en tomografías; se ignora.
        """
        if self._data.ndim == 2:
            corte = self._data.copy()
            indice_label = "2D"
        else:
            if slice_index is None:
                slice_index = self._data.shape[0] // 2
            corte = self.obtener_corte(slice_index, "axial")
            indice_label = f"axial #{slice_index}"

        corte_w = self._aplicar_ventana(corte)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            corte_w,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal"
        )
        plt.colorbar(im, ax=ax, label="Intensidad (ventana)")
        ax.set_title(
            f"Tomografía — corte {indice_label}\n"
            f"WC={self._window_center:.0f}  "
            f"WW={self._window_width:.0f}"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # REGISTRAR TRANSFORMACIÓN
    # =========================================================
    def registrar_transformacion(
        self,
        descripcion: str
    ) -> None:
        """
        Agrega manualmente una entrada al historial.
        """
        if not isinstance(descripcion, str):
            raise TypeError(
                "descripcion debe ser string"
            )
        if not descripcion.strip():
            raise ValueError("descripcion vacía")

        self._info["historial"].modificar_historial(
            descripcion
        )

    # =========================================================
    # DIMENSIONES FÍSICAS
    # =========================================================
    def dimensiones_fisicas(self) -> dict:
        """
        Retorna las dimensiones físicas del volumen en mm.

        Retorna
        -------
        dict con claves:
          - "slices_mm"   : extensión total en el eje axial
          - "filas_mm"    : extensión total en filas
          - "columnas_mm" : extensión total en columnas
          - "voxel_mm3"   : volumen de un voxel en mm³
        """
        if self._data.ndim == 3:
            n_slices, n_filas, n_cols = self._data.shape
        else:
            n_slices = 1
            n_filas, n_cols = self._data.shape

        sz, sy, sx = self._voxel_spacing
        return {
            "slices_mm":   n_slices * sz,
            "filas_mm":    n_filas  * sy,
            "columnas_mm": n_cols   * sx,
            "voxel_mm3":   sz * sy * sx,
        }

    # =========================================================
    # REPRESENTACIÓN
    # =========================================================
    def __str__(self) -> str:
        return (
            f"ImagenTomografia("
            f"shape={self._data.shape}, "
            f"escala='{self._escala_intensidad}', "
            f"WC={self._window_center:.0f}, "
            f"WW={self._window_width:.0f}, "
            f"voxel={self._voxel_spacing})"
        )

    # =========================================================
    # EXPLORADOR INTERACTIVO DE SLICES (PLOTLY)
    # =========================================================
    def explorar_slices(
        self,
        eje: str = "axial",
        center: float = None,
        width: float = None,
        colorscale: str = "gray",
    ) -> None:
        """
        Abre un explorador interactivo de slices en el navegador
        usando Plotly. Permite recorrer todos los cortes del
        volumen mediante un slider.

        El explorador muestra:
        - El corte actual con windowing aplicado.
        - Un slider para navegar entre slices.
        - Botones de reproducción automática (play/pause).
        - Información del corte (índice, WC, WW) en el título.

        Parámetros
        ----------
        eje : str
            "axial", "coronal" o "sagital". Default: "axial".
        center : float, opcional
            Window center en HU. Default: self._window_center.
        width : float, opcional
            Window width en HU. Default: self._window_width.
        colorscale : str, opcional
            Escala de color de Plotly.
            Opciones útiles: "gray", "hot", "viridis", "jet".
            Default: "gray".

        Notas
        -----
        Requiere plotly instalado:
            pip install plotly
        Abre el resultado en el navegador por defecto.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly no está instalado. "
                "Ejecutá: pip install plotly"
            )

        if self._data.ndim != 3:
            raise ValueError(
                "explorar_slices requiere volumen 3-D"
            )

        ejes_validos = ("axial", "coronal", "sagital")
        if eje not in ejes_validos:
            raise ValueError(
                f"eje debe ser uno de {ejes_validos}"
            )

        if center is None:
            center = self._window_center
        if width is None:
            width = self._window_width

        # Determinar cantidad de slices según el eje
        eje_idx = {"axial": 0, "coronal": 1, "sagital": 2}[eje]
        n = self._data.shape[eje_idx]

        # --------------------------------------------------
        # PRE-PROCESAR TODOS LOS SLICES
        # Aplicar windowing a cada corte y convertir a uint8
        # para reducir el tamaño de los datos en el HTML
        # --------------------------------------------------
        slices_procesados = []
        for i in range(n):
            corte = self.obtener_corte(i, eje)
            corte_w = self._aplicar_ventana(
                corte, center, width
            ).astype(np.uint8)
            slices_procesados.append(corte_w)

        # --------------------------------------------------
        # CONSTRUIR FRAMES DE PLOTLY
        # Cada frame es un corte del volumen
        # --------------------------------------------------
        frames = []
        for i, corte_w in enumerate(slices_procesados):
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=corte_w,
                        colorscale=colorscale,
                        zmin=0,
                        zmax=255,
                        showscale=True,
                        colorbar=dict(
                            title="HU (ventana)",
                            thickness=15,
                            len=0.8,
                        ),
                    )
                ],
                name=str(i),
                layout=go.Layout(
                    title=dict(
                        text=(
                            f"Tomografía CT — eje {eje} | "
                            f"Corte {i + 1}/{n} | "
                            f"WC={center:.0f}  WW={width:.0f}"
                        )
                    )
                )
            )
            frames.append(frame)

        # --------------------------------------------------
        # FIGURA INICIAL (primer slice)
        # --------------------------------------------------
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=slices_procesados[0],
                    colorscale=colorscale,
                    zmin=0,
                    zmax=255,
                    showscale=True,
                    colorbar=dict(
                        title="HU (ventana)",
                        thickness=15,
                        len=0.8,
                    ),
                )
            ],
            frames=frames,
        )

        # --------------------------------------------------
        # SLIDER — un paso por slice
        # --------------------------------------------------
        pasos_slider = []
        for i in range(n):
            paso = dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(i),
            )
            pasos_slider.append(paso)

        slider = dict(
            active=0,
            steps=pasos_slider,
            currentvalue=dict(
                prefix=f"Corte {eje}: ",
                visible=True,
                xanchor="center",
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.05,
            y=0,
        )

        # --------------------------------------------------
        # BOTONES PLAY / PAUSE
        # --------------------------------------------------
        botones = [
            dict(
                label="▶ Play",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(
                            duration=120,
                            redraw=True
                        ),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    ),
                ],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(
                            duration=0,
                            redraw=False
                        ),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ]

        # --------------------------------------------------
        # LAYOUT FINAL
        # --------------------------------------------------
        fig.update_layout(
            title=dict(
                text=(
                    f"Tomografía CT — eje {eje} | "
                    f"Corte 1/{n} | "
                    f"WC={center:.0f}  WW={width:.0f}"
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=14),
            ),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                scaleanchor="x",   # mantiene proporción real
                scaleratio=1,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=botones,
                    x=0.05,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                    bgcolor="lightgray",
                    bordercolor="gray",
                    font=dict(size=12),
                )
            ],
            sliders=[slider],
            margin=dict(l=20, r=20, t=80, b=80),
            height=650,
            plot_bgcolor="black",
            paper_bgcolor="white",
        )

        fig.show()

        self._info["historial"].modificar_historial(
            f"Explorador interactivo de slices — eje {eje} "
            f"({n} cortes, WC={center:.0f}, WW={width:.0f})"
        )

