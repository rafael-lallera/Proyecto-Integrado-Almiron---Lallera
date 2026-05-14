import numpy as np
import matplotlib.pyplot as plt
from bioimagenes.core.info import Info


class Imagen:

    # -------------------------
    # CONSTRUCTOR
    # -------------------------
    def __init__(self, data: np.ndarray, info: Info):

        # Validar tipo
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un numpy array")

        # Validar dimensiones (2D o 3D)
        if data.ndim != 2 and data.ndim != 3:
            raise ValueError("La imagen debe ser 2D o 3D")

        # Validar info
        if not isinstance(info, Info):
            raise TypeError("info debe ser un objeto Info")

        # Validar coherencia dimensiones
        if info["dimensiones"] != data.shape:
            raise ValueError(
                f"Dimensiones inconsistente: {info['dimensiones']} vs {data.shape}"
            )

        self._data = data
        self._info = info

    # -------------------------
    # APLICAR FILTRO
    # -------------------------
    def aplicar_filtro(self, filtro):

        if not hasattr(filtro, "aplicar_filtro"):
            raise TypeError("Filtro inválido")

        self._data = filtro.aplicar_filtro(self._data)

        self._info["historial"].modificar_historial(
            "Filtro aplicado"
        )

    # -------------------------
    # BLANCO Y NEGRO
    # -------------------------
    def bn(self):

        if self._data.ndim == 3:
            self._data = np.mean(self._data, axis=2)

            self._info["historial"].modificar_historial(
                "Conversión a blanco y negro"
            )
        else:
            print("La imagen ya es 2D")

    # -------------------------
    # VISUALIZAR
    # -------------------------
    def visualizar(self, slice_index=None):

        plt.figure()

        # -------------------------
        # CASO 2D → GRIS
        # -------------------------
        if self._data.ndim == 2:
            plt.imshow(self._data, cmap="gray")

        # -------------------------
        # CASO RGB (3 canales)
        # -------------------------
        elif self._data.ndim == 3 and self._data.shape[2] == 3:
            plt.imshow(self._data)

        # -------------------------
        # CASO RGBA (4 canales)
        # -------------------------
        elif self._data.ndim == 3 and self._data.shape[2] == 4:
            plt.imshow(self._data)

        # -------------------------
        # CASO VOLUMEN (ej: tomografía)
        # -------------------------
        elif self._data.ndim == 3:

            if slice_index is None:
                slice_index = self._data.shape[2] // 2

            plt.imshow(self._data[:, :, slice_index], cmap="gray")
            plt.title(f"Slice {slice_index}")

        else:
            raise ValueError("Formato de imagen no soportado")

        plt.axis("off")
        plt.show()

    # -------------------------
    # NORMALIZAR
    # -------------------------
    def normalizar(self):

        min_val = np.min(self._data)
        max_val = np.max(self._data)

        if max_val - min_val == 0:
            raise ValueError("Imagen constante")

        self._data = (self._data - min_val) / (max_val - min_val)

        self._info["historial"].modificar_historial(
            "Normalización aplicada"
        )

    # -------------------------
    # MÉTODOS ESPECIALES
    # -------------------------
    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, key):
        return self._data[key]

    def __str__(self):
        return f"Imagen(shape={self._data.shape}, dtype={self._data.dtype})"
    
    # -------------------------
    # MÉTODOS de RECORTE
    # -------------------------
    
    def recortar(self, x_ini, x_fin, y_ini, y_fin):

        # -------------------------
        # VALIDACIONES
        # -------------------------
        if not all(isinstance(v, int) for v in [x_ini, x_fin, y_ini, y_fin]):
            raise TypeError("Las coordenadas deben ser enteros")

        if x_ini < 0 or y_ini < 0:
            raise ValueError("Índices negativos no permitidos")

        if x_fin > self._data.shape[0] or y_fin > self._data.shape[1]:
            raise ValueError("El recorte excede dimensiones")

        if x_ini >= x_fin or y_ini >= y_fin:
            raise ValueError("Rangos inválidos")

        # -------------------------
        # RECORTE
        # -------------------------
        if self._data.ndim == 2:
            nueva_data = self._data[x_ini:x_fin, y_ini:y_fin]

        elif self._data.ndim == 3:
            nueva_data = self._data[x_ini:x_fin, y_ini:y_fin, :]

        else:
            raise ValueError("Dimensión no soportada")

        # -------------------------
        # NUEVO HISTORIAL
        # -------------------------
        from bioimagenes.core.historial import Historial
        nuevo_historial = Historial()

        ruta = self._info["ruta_origen"]

        mensaje = "Imagen generada a partir de un recorte"

        if ruta is not None:
            mensaje += f" de la imagen original: {ruta}"

        nuevo_historial.modificar_historial(mensaje)

        # -------------------------
        # NUEVO INFO
        # -------------------------
        from bioimagenes.core.info import Info

        nueva_info = Info(
            dimensiones=nueva_data.shape,
            brillo=self._info["brillo"],
            historial=nuevo_historial,
            cortada=True,
            ruta_origen=ruta  
        )

        # -------------------------
        # REGISTRAR EN ORIGINAL
        # -------------------------
        self._info["historial"].modificar_historial(
            f"Recorte aplicado: ({x_ini}:{x_fin}, {y_ini}:{y_fin})"
        )

        # -------------------------
        # CREAR NUEVA IMAGEN
        # -------------------------
        nueva_imagen = Imagen(nueva_data, nueva_info)

        return nueva_imagen