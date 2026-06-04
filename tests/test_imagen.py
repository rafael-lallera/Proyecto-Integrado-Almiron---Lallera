"""
tests/test_recorte_imagen.py

Prueba de funcionamiento del método recortar() de la clase Imagen.

Objetivos
---------
1. Cargar una imagen RGB desde disco.
2. Crear un objeto Imagen con sus metadatos asociados.
3. Aplicar un recorte rectangular.
4. Verificar que la nueva imagen conserve sus dimensiones.
5. Visualizar la imagen original y la recortada.
6. Comprobar la trazabilidad mediante el historial y la
   ruta de origen almacenada en Info.

Dependencias
------------
- numpy
- matplotlib.pyplot
- bioimagenes.core.imagen
- bioimagenes.core.info
"""

import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info


# =========================================================
# CARGA DE IMAGEN
# =========================================================
#
# Se carga una imagen RGB de ejemplo para utilizarla
# durante la prueba.
#
ruta = "docs/examples/Imagen de prueba RGB.jpg"

img_array = plt.imread(ruta)

#
# Algunas imágenes pueden incluir un canal alfa
# (RGBA). Para esta prueba se conserva únicamente
# la información RGB.
#
if img_array.ndim == 3 and img_array.shape[2] == 4:
    img_array = img_array[:, :, :3]

print("Imagen cargada correctamente")
print(f"Dimensiones: {img_array.shape}")


# =========================================================
# CREACIÓN DEL OBJETO IMAGEN
# =========================================================
#
# Se construyen los metadatos asociados a la imagen
# y posteriormente el objeto Imagen.
#
info = Info(
    dimensiones=img_array.shape,
    brillo=1.0,
    ruta_origen=ruta
)

img = Imagen(img_array, info)

print("\n=== IMAGEN ORIGINAL ===")
print(img)


# =========================================================
# APLICACIÓN DEL RECORTE
# =========================================================
#
# Se extrae una región rectangular comprendida entre:
#
# Filas:    50 -> 200
# Columnas: 50 -> 200
#
print("\n=== APLICANDO RECORTE ===")

img_recortada = img.recortar(
    x_ini=50,
    x_fin=200,
    y_ini=50,
    y_fin=200
)

print("\n=== IMAGEN RECORTADA ===")
print(img_recortada)


# =========================================================
# VISUALIZACIÓN
# =========================================================
#
# Se muestran ambas imágenes para comprobar visualmente
# que el recorte fue realizado correctamente.
#
print("\nMostrando imagen original...")

#
# La visualización se fuerza en modo RGB para evitar
# que una imagen de tres canales sea interpretada como
# un volumen tridimensional.
#
img.visualizar(modo="rgb")

print("\nMostrando imagen recortada...")

img_recortada.visualizar(modo="rgb")


# =========================================================
# HISTORIAL DE OPERACIONES
# =========================================================
#
# Se verifica que la operación de recorte quede
# registrada correctamente tanto en la imagen original
# como en la imagen derivada.
#
print("\n=== HISTORIAL ORIGINAL ===")
print(info["historial"])

print("\n=== HISTORIAL IMAGEN RECORTADA ===")
print(img_recortada._info["historial"])


# =========================================================
# VALIDACIÓN DE TRAZABILIDAD
# =========================================================
#
# Ambas imágenes deben conservar la referencia a la
# imagen fuente original.
#
print("\n=== VALIDACIÓN DE TRAZABILIDAD ===")

print("\nRuta almacenada en la imagen original:")
print(info["ruta_origen"])

print("\nRuta almacenada en la imagen recortada:")
print(img_recortada._info["ruta_origen"])


# =========================================================
# VALIDACIÓN DE DIMENSIONES
# =========================================================
#
# El recorte definido entre 50 y 200 produce una imagen
# de 150 x 150 píxeles.
#
print("\n=== VALIDACIÓN DE DIMENSIONES ===")

print("Dimensiones imagen original:")
print(img_array.shape)

print("\nDimensiones imagen recortada:")
print(img_recortada._data.shape)