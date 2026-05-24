"""
tests/test_manual_termografia.py

Prueba manual de la clase ImagenTermografica utilizando
una imagen termográfica real cargada desde disco.

Descripción general
-------------------
Este script permite validar manualmente el funcionamiento de la
clase ImagenTermografica mediante una secuencia completa de pruebas
sobre una imagen real.

El flujo implementado incluye:

- Carga de la imagen termográfica.
- Conversión a escala de grises.
- Conversión del tipo de dato a float64.
- Creación del objeto Info.
- Creación del objeto ImagenTermografica.
- Visualización de la imagen original.
- Interpretación de temperaturas.
- Obtención del rango térmico.
- Detección de puntos calientes.
- Segmentación por umbral.
- Conversión de unidades de temperatura.
- Normalización de la imagen.
- Visualización de resultados.
- Consulta del historial de operaciones.

Objetivo
--------
Verificar de forma visual y funcional que los métodos de la clase
ImagenTermografica operen correctamente sobre datos reales.

Dependencias
------------
- numpy
- matplotlib
- bioimagenes.core.info
- bioimagenes.medicas.imagen_termografica

Autor
-----
Proyecto Bioimágenes

"""

import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.info import Info
from bioimagenes.medicas.imagen_termografica import ImagenTermografica


# ============================================================
# 1. CARGAR IMAGEN
# ============================================================
#
# Se carga desde disco una imagen termográfica real utilizando
# matplotlib. Luego se muestra información básica asociada
# al tamaño y estructura de la imagen.
#

ruta = "docs/termografias/mate.jpg"

img = plt.imread(ruta)

print("Imagen cargada correctamente")

print("Shape original:")
print(img.shape)


# ============================================================
# 2. CONVERTIR A ESCALA DE GRISES
# ============================================================
#
# Si la imagen posee múltiples canales (RGB),
# se convierte a escala de grises calculando el promedio
# de los tres canales de color.
#

if img.ndim == 3:

    img = np.mean(img[:, :, :3], axis=2)

print("\nImagen convertida a grayscale")

print("Nuevo shape:")
print(img.shape)


# ============================================================
# 3. CONVERTIR A FLOAT64
# ============================================================
#
# Se convierte la imagen al tipo float64 para permitir
# operaciones numéricas de mayor precisión durante el
# procesamiento térmico.
#

data_temp = img.astype(np.float64)

print("\nTipo de dato:")
print(data_temp.dtype)


# ============================================================
# 4. CREAR INFO
# ============================================================
#
# Se crea un objeto Info que almacena metadatos asociados
# a la imagen, como dimensiones, brillo y ruta de origen.
#

info = Info(
    dimensiones=data_temp.shape,
    brillo=1.0,
    ruta_origen=ruta
)

print("\nObjeto Info creado")


# ============================================================
# 5. CREAR IMAGEN TERMOGRÁFICA
# ============================================================
#
# Se instancia un objeto ImagenTermografica utilizando:
#
# - Datos térmicos de la imagen.
# - Información auxiliar.
# - Unidad de temperatura.
# - Rango térmico mínimo y máximo.
# - Parámetros de escala.
# - Umbral térmico basado en la media de temperaturas.
#

termo = ImagenTermografica(
    data_temp,
    info,
    unidad="C",
    rango_temp=(
        float(data_temp.min()),
        float(data_temp.max())
    ),
    escala=(1.0, 0.0),
    umbral_calor=float(np.mean(data_temp))
)

print("\nObjeto ImagenTermografica creado")

print(termo)


# ============================================================
# 6. VISUALIZAR IMAGEN ORIGINAL
# ============================================================
#
# Se visualiza la imagen térmica original utilizando
# una paleta de colores pseudo-térmica.
#

plt.figure(figsize=(6, 6))

plt.imshow(data_temp, cmap="RdBu_r")

plt.title("Imagen Termográfica Original")

plt.colorbar(label="Temperatura")

plt.show()


# ============================================================
# 7. INTERPRETAR TEMPERATURA
# ============================================================
#
# Se interpreta la información térmica contenida en la imagen
# y se muestran los valores mínimo y máximo detectados.
#

temperaturas = termo.interpretar_temperatura()

print("\nTemperaturas interpretadas")

print("Mínimo:")
print(temperaturas.min())

print("Máximo:")
print(temperaturas.max())


# ============================================================
# 8. OBTENER RANGO
# ============================================================
#
# Se obtiene el rango térmico configurado para la imagen.
#

rango = termo.obtener_rango()

print("\nRango térmico:")

print(rango)


# ============================================================
# 9. DETECTAR PUNTOS CALIENTES
# ============================================================
#
# Se genera una máscara binaria que identifica las regiones
# cuya temperatura supera el umbral térmico definido.
#

mascara = termo.detectar_puntos_calientes()

print("\nMáscara de puntos calientes creada")

print("Shape:")
print(mascara.shape)


# ============================================================
# 10. VISUALIZAR PUNTOS CALIENTES
# ============================================================
#
# Se visualiza la máscara binaria correspondiente
# a los puntos calientes detectados.
#

plt.figure(figsize=(6, 6))

plt.imshow(mascara, cmap="gray")

plt.title("Puntos Calientes")

plt.show()


# ============================================================
# 11. SEGMENTAR POR UMBRAL
# ============================================================
#
# Se realiza una segmentación de la imagen utilizando
# el umbral térmico configurado.
#

segmentada = termo.segmentar_por_umbral()

print("\nImagen segmentada creada")


# ============================================================
# 12. VISUALIZAR SEGMENTACIÓN
# ============================================================
#
# Se visualiza el resultado de la segmentación binaria.
#

plt.figure(figsize=(6, 6))

plt.imshow(segmentada, cmap="gray")

plt.title("Segmentación por Umbral")

plt.show()


# ============================================================
# 13. CONVERTIR A KELVIN
# ============================================================
#
# Se convierte la representación térmica desde grados Celsius
# hacia Kelvin.
#

kelvin = termo.convertir_a_temperatura("K")

print("\nConversión a Kelvin")

print(kelvin)


# ============================================================
# 14. NORMALIZAR
# ============================================================
#
# Se normalizan los datos de la imagen térmica al rango
# estándar definido por la implementación.
#

termo.normalizar()

print("\nImagen normalizada")

print("Nuevo mínimo:")
print(termo._data.min())

print("Nuevo máximo:")
print(termo._data.max())


# ============================================================
# 15. VISUALIZAR NORMALIZADA
# ============================================================
#
# Se visualiza la imagen luego del proceso de normalización.
#

plt.figure(figsize=(6, 6))

plt.imshow(termo._data, cmap="RdBu_r")

plt.title("Imagen Normalizada")

plt.colorbar()

plt.show()


# ============================================================
# 16. HISTORIAL
# ============================================================
#
# Se imprime el historial de operaciones almacenado
# dentro del objeto Info.
#

print("\nHistorial:")

print(info["historial"])