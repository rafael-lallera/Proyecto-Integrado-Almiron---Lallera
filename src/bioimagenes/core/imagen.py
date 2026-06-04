"""
bioimagenes/core/imagen.py

Clase base para representar imágenes dentro del framework
de bioimágenes.

Descripción general
-------------------
La clase Imagen encapsula una representación matricial de una
imagen utilizando numpy arrays y administra metadatos mediante
la clase Info.

La clase soporta:

- Imágenes 2-D en escala de grises.
- Imágenes RGB/RGBA.
- Volúmenes 3-D.
- Aplicación de filtros.
- Conversión a blanco y negro.
- Visualización.
- Normalización.
- Recorte de regiones.

Además, todas las operaciones relevantes quedan registradas
en el historial asociado al objeto Info.

Dependencias
------------
- numpy
- matplotlib.pyplot
- bioimagenes.core.info

Autor
-----
Proyecto Bioimágenes
"""

import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.info import Info


class Imagen:
    """
    Clase base para representar imágenes digitales.

    La imagen se almacena internamente como un numpy array
    y se acompaña de un objeto Info que contiene metadatos
    asociados.

    Atributos
    ---------
    _data : np.ndarray
        Datos numéricos de la imagen.

    _info : Info
        Metadatos asociados a la imagen.
    """

    # =========================================================
    # CONSTRUCTOR
    # =========================================================
    def __init__(self, data: np.ndarray, info: Info):
        """
        Inicializa una imagen.

        Parámetros
        ----------
        data : np.ndarray
            Datos de la imagen.

        info : Info
            Objeto con metadatos asociados.

        Excepciones
        -----------
        TypeError
            Si los tipos de entrada no son válidos.

        ValueError
            Si las dimensiones no son compatibles.
        """

        # -----------------------------------------------------
        # VALIDAR TIPO
        # -----------------------------------------------------
        if not isinstance(data, np.ndarray):
            raise TypeError("data debe ser un numpy array")

        # -----------------------------------------------------
        # VALIDAR DIMENSIONES
        # -----------------------------------------------------
        #
        # La clase soporta:
        #
        # - Imágenes 2-D.
        # - Imágenes RGB/RGBA.
        # - Volúmenes 3-D.
        #
        if data.ndim not in (2, 3):
            raise ValueError("La imagen debe ser 2D o 3D")

        # -----------------------------------------------------
        # VALIDAR INFO
        # -----------------------------------------------------
        if not isinstance(info, Info):
            raise TypeError("info debe ser un objeto Info")

        # -----------------------------------------------------
        # VALIDAR COHERENCIA DIMENSIONAL
        # -----------------------------------------------------
        if info["dimensiones"] != data.shape:
            raise ValueError(
                f"Dimensiones inconsistentes: "
                f"{info['dimensiones']} vs {data.shape}"
            )

        # -----------------------------------------------------
        # CONVERSIÓN INTERNA
        # -----------------------------------------------------
        #
        # Se fuerza float64 para mantener consistencia
        # numérica en todo el framework.
        #
        self._data = data.astype(np.float64)

        self._info = info

    # =========================================================
    # APLICAR FILTRO
    # =========================================================
    def aplicar_filtro(self, filtro) -> None:
        """
        Aplica un filtro sobre la imagen.

        Parámetros
        ----------
        filtro : objeto
            Debe implementar el método aplicar_filtro(data).

        Excepciones
        -----------
        TypeError
            Si el filtro no es válido.
        """

        if not hasattr(filtro, "aplicar_filtro"):
            raise TypeError("Filtro inválido")

        self._data = filtro.aplicar_filtro(self._data)

        self._info["historial"].modificar_historial(
            f"Filtro aplicado: {filtro}"
        )

    # =========================================================
    # CONVERSIÓN A BLANCO Y NEGRO
    # =========================================================
    def bn(self) -> None:
        """
        Convierte una imagen RGB/RGBA a escala de grises.

        La conversión se realiza promediando los canales
        de color.

        Notas
        -----
        - Si la imagen ya es 2-D, no se realiza ninguna acción.
        - El método actualiza automáticamente las dimensiones
          almacenadas en Info.
        """

        # -----------------------------------------------------
        # IMAGEN YA EN ESCALA DE GRISES
        # -----------------------------------------------------
        if self._data.ndim == 2:

            self._info["historial"].modificar_historial(
                "Conversión BN omitida: imagen ya 2D"
            )

            return

        # -----------------------------------------------------
        # VALIDAR FORMATO MULTICANAL
        # -----------------------------------------------------
        #
        # Se aceptan imágenes RGB o RGBA.
        #
        if self._data.shape[2] not in (3, 4):
            raise ValueError(
                "La conversión BN solo es válida para imágenes RGB/RGBA"
            )

        # -----------------------------------------------------
        # CONVERSIÓN
        # -----------------------------------------------------
        #
        # Se ignora el canal alfa en imágenes RGBA.
        #
        self._data = np.mean(
            self._data[:, :, :3],
            axis=2
        ).astype(np.float64)

        # -----------------------------------------------------
        # ACTUALIZAR METADATOS
        # -----------------------------------------------------
        self._info["dimensiones"] = self._data.shape

        self._info["historial"].modificar_historial(
            "Conversión a blanco y negro"
        )

    # =========================================================
    # VISUALIZAR
    # =========================================================
    def visualizar(
        self,
        slice_index: int = None,
        modo: str = None
    ) -> None:
        """
        Visualiza la imagen utilizando matplotlib.

        Parámetros
        ----------
        slice_index : int, opcional
            Slice a visualizar en volúmenes 3-D.

        modo : str, opcional
            Permite especificar explícitamente el tipo de imagen:
            - "rgb"
            - "rgba"
            - "volumen"

        Excepciones
        -----------
        ValueError
            Si el formato de imagen no es válido.
        """

        plt.figure(figsize=(6, 6))

        # =====================================================
        # CASO 2D
        # =====================================================
        if self._data.ndim == 2:

            plt.imshow(self._data, cmap="gray")

        # =====================================================
        # CASO 3D
        # =====================================================
        elif self._data.ndim == 3:

            # -------------------------------------------------
            # INFERENCIA AUTOMÁTICA DEL MODO
            # -------------------------------------------------
            #
            # Si el usuario no especifica el modo,
            # se intenta determinar automáticamente
            # el tipo de imagen a partir del número
            # de canales.
            #
                if modo is None:

                    if self._data.shape[2] == 3:
                        modo = "rgb"

                    elif self._data.shape[2] == 4:
                        modo = "rgba"

                    else:
                        modo = "volumen"

                # -------------------------------------------------
                # RGB
                # -------------------------------------------------
                if modo == "rgb":

                    if self._data.shape[2] != 3:
                        raise ValueError(
                            "La imagen no posee 3 canales RGB"
                        )

                    img = self._data.copy()

                    #
                    # Matplotlib espera imágenes RGB float
                    # en rango [0,1].
                    #
                    if img.max() > 1.0:
                        img = img / 255.0

                    plt.imshow(img)

                # -------------------------------------------------
                # RGBA
                # -------------------------------------------------
                elif modo == "rgba":

                    if self._data.shape[2] != 4:
                        raise ValueError(
                            "La imagen no posee 4 canales RGBA"
                        )

                    img = self._data.copy()

                    #
                    # Matplotlib espera imágenes RGBA float
                    # en rango [0,1].
                    #
                    if img.max() > 1.0:
                        img = img / 255.0

                    plt.imshow(img)

                # -------------------------------------------------
                # VOLUMEN
                # -------------------------------------------------
                elif modo == "volumen":

                    # ---------------------------------------------
                    # VALIDAR ÍNDICE
                    # ---------------------------------------------
                    if slice_index is None:
                        slice_index = self._data.shape[2] // 2

                    if not isinstance(slice_index, int):
                        raise TypeError(
                            "slice_index debe ser entero"
                        )

                    if (
                        slice_index < 0
                        or slice_index >= self._data.shape[2]
                    ):
                        raise ValueError(
                            "slice_index fuera de rango"
                        )

                    plt.imshow(
                        self._data[:, :, slice_index],
                        cmap="gray"
                    )

                    plt.title(f"Slice {slice_index}")

                # -------------------------------------------------
                # MODO INVÁLIDO
                # -------------------------------------------------
                else:
                    raise ValueError(
                        "modo debe ser 'rgb', 'rgba' o 'volumen'"
                    )

        else:
            raise ValueError("Formato de imagen no soportado")

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # NORMALIZAR
    # =========================================================
    def normalizar(self) -> None:
        """
        Normaliza la imagen al rango [0, 1].

        Excepciones
        -----------
        ValueError
            Si la imagen es constante.
        """

        min_val = float(np.min(self._data))
        max_val = float(np.max(self._data))

        if max_val - min_val == 0:
            raise ValueError("Imagen constante")

        self._data = (
            (self._data - min_val)
            / (max_val - min_val)
        ).astype(np.float64)

        self._info["historial"].modificar_historial(
            "Normalización aplicada"
        )

    # =========================================================
    # RECORTE
    # =========================================================
    def recortar(
        self,
        x_ini: int,
        x_fin: int,
        y_ini: int,
        y_fin: int
    ) -> "Imagen":
        """
        Genera una nueva imagen recortada.

        Parámetros
        ----------
        x_ini : int
            Fila inicial.

        x_fin : int
            Fila final.

        y_ini : int
            Columna inicial.

        y_fin : int
            Columna final.

        Retorna
        -------
        Imagen
            Nueva imagen recortada.

        Excepciones
        -----------
        TypeError
            Si las coordenadas no son enteras.

        ValueError
            Si los índices son inválidos.
        """

        # -----------------------------------------------------
        # VALIDAR TIPOS
        # -----------------------------------------------------
        if not all(
            isinstance(v, int)
            for v in [x_ini, x_fin, y_ini, y_fin]
        ):
            raise TypeError("Las coordenadas deben ser enteros")

        # -----------------------------------------------------
        # VALIDAR RANGOS
        # -----------------------------------------------------
        if x_ini < 0 or y_ini < 0:
            raise ValueError("Índices negativos no permitidos")

        if x_ini >= x_fin or y_ini >= y_fin:
            raise ValueError("Rangos inválidos")

        if (
            x_fin > self._data.shape[0]
            or y_fin > self._data.shape[1]
        ):
            raise ValueError("El recorte excede dimensiones")

        # -----------------------------------------------------
        # RECORTE 2D
        # -----------------------------------------------------
        if self._data.ndim == 2:

            nueva_data = self._data[
                x_ini:x_fin,
                y_ini:y_fin
            ].copy()

        # -----------------------------------------------------
        # RECORTE 3D
        # -----------------------------------------------------
        elif self._data.ndim == 3:

            nueva_data = self._data[
                x_ini:x_fin,
                y_ini:y_fin,
                :
            ].copy()

        else:
            raise ValueError("Dimensión no soportada")

        # -----------------------------------------------------
        # NUEVO HISTORIAL
        # -----------------------------------------------------
        from bioimagenes.core.historial import Historial

        nuevo_historial = Historial()

        ruta = self._info["ruta_origen"]

        mensaje = "Imagen generada a partir de un recorte"

        if ruta is not None:
            mensaje += f" de la imagen original: {ruta}"

        nuevo_historial.modificar_historial(mensaje)

        # -----------------------------------------------------
        # NUEVO INFO
        # -----------------------------------------------------
        nueva_info = Info(
            dimensiones=nueva_data.shape,
            brillo=self._info["brillo"],
            historial=nuevo_historial,
            cortada=True,
            ruta_origen=ruta
        )

        # -----------------------------------------------------
        # REGISTRAR EN ORIGINAL
        # -----------------------------------------------------
        self._info["historial"].modificar_historial(
            f"Recorte aplicado: "
            f"({x_ini}:{x_fin}, {y_ini}:{y_fin})"
        )

        # -----------------------------------------------------
        # CREAR NUEVA IMAGEN
        # -----------------------------------------------------
        return Imagen(nueva_data, nueva_info)

    # =========================================================
    # MÉTODOS ESPECIALES
    # =========================================================
    def __len__(self) -> int:
        """
        Retorna la cantidad de filas de la imagen.
        """
        return self._data.shape[0]

    def __getitem__(self, key):
        """
        Permite indexar directamente sobre la imagen.
        """
        return self._data[key]

    def __str__(self) -> str:
        """
        Retorna una representación textual resumida.

        Retorna
        -------
        str
            Descripción de la imagen.
        """
        return (
            f"Imagen(shape={self._data.shape}, "
            f"dtype={self._data.dtype})"
        )