"""
tests/test_manual_radiografia.py

Prueba manual de la clase ImagenRadiografia utilizando
una radiografía real cargada desde disco.

Descripción general
-------------------
Este script permite validar manualmente el funcionamiento
de la clase ImagenRadiografia mediante una secuencia de
pruebas visuales y funcionales.

El objetivo es comprobar:

- La correcta carga de la imagen.
- La conversión a escala de grises.
- La creación de objetos Info e ImagenRadiografia.
- La visualización de radiografías.
- La selección de regiones de interés (ROI).
- El ajuste de visualización.
- La mejora de contraste.
- La inversión de intensidades.
- La ecualización del histograma.
- La detección de bordes mediante Sobel.
- El clustering de intensidades.
- La aplicación de filtros.
- El funcionamiento del historial.

Cada etapa incluye comentarios explicativos indicando
qué se busca validar y qué comportamiento se espera.

Dependencias
------------
- numpy
- matplotlib
- bioimagenes.core.info
- bioimagenes.medicas.imagen_radiografia
- bioimagenes.filtros.filtro

Autor
-----
Proyecto Bioimágenes
"""

import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.info import Info
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia
from bioimagenes.filtros.filtro import (
    FiltroGaussiano,
    FiltroNitidez
)


def buscar_roi_interesante(data: np.ndarray, alto: int = 200, ancho: int = 200, paso: int = 80):
    """
    Busca una ROI con mayor variación de intensidades dentro de
    una región central de la imagen.

    La idea es evitar zonas homogéneas o de fondo, que suelen
    verse completamente negras o blancas al visualizarse.

    Parámetros
    ----------
    data : np.ndarray
        Imagen 2-D de entrada.

    alto : int
        Alto de la ventana a recortar.

    ancho : int
        Ancho de la ventana a recortar.

    paso : int
        Paso de búsqueda entre ventanas.

    Retorna
    -------
    tuple[int, int, float]
        Coordenadas (x, y) de la mejor ROI y su desviación estándar.
    """
    filas, columnas = data.shape

    # Se restringe la búsqueda a la zona central para evitar
    # bordes vacíos o zonas externas de la placa.
    fila_min = int(filas * 0.15)
    fila_max = int(filas * 0.85) - alto
    col_min = int(columnas * 0.15)
    col_max = int(columnas * 0.85) - ancho

    if fila_max < fila_min:
        fila_min = 0
        fila_max = filas - alto

    if col_max < col_min:
        col_min = 0
        col_max = columnas - ancho

    mejor_x = fila_min
    mejor_y = col_min
    mejor_score = -np.inf

    for x in range(fila_min, fila_max + 1, paso):
        for y in range(col_min, col_max + 1, paso):
            ventana = data[x:x + alto, y:y + ancho]

            if ventana.shape != (alto, ancho):
                continue

            score = float(ventana.std())

            if score > mejor_score:
                mejor_score = score
                mejor_x = x
                mejor_y = y

    return mejor_x, mejor_y, mejor_score


# ============================================================
# 1. CARGA DE IMAGEN
# ============================================================
#
# Se carga una radiografía real desde disco utilizando
# matplotlib.
#
# Se espera:
# - Que la imagen se cargue correctamente.
# - Que pueda visualizarse su shape.
# - Que los datos sean compatibles con numpy.
#

ruta = r"docs/radiografias/sample/46523715740384360192496023767246369337_veyewt.png"

img = plt.imread(ruta)

print("Radiografía cargada correctamente")
print("Shape original:")
print(img.shape)
print("Tipo de dato original:")
print(img.dtype)

# ============================================================
# 2. CONVERSIÓN A ESCALA DE GRISES
# ============================================================
#
# Si la imagen es RGB o RGBA, se convierte a escala
# de grises promediando los primeros tres canales.
#
# Se espera:
# - Obtener una matriz 2-D.
# - Mantener la información estructural de la radiografía.
#

if img.ndim == 3:
    img = np.mean(img[:, :, :3], axis=2)

print("\nConversión a escala de grises completada")
print("Nuevo shape:")
print(img.shape)

# ============================================================
# 3. CONVERSIÓN A FLOAT64
# ============================================================
#
# La arquitectura del framework trabaja internamente
# utilizando float64 para mantener consistencia numérica.
#
# Se espera:
# - Obtener un array float64.
#

data_rx = img.astype(np.float64)

print("\nConversión a float64 realizada")
print("Nuevo dtype:")
print(data_rx.dtype)

# ============================================================
# 4. CREACIÓN DEL OBJETO INFO
# ============================================================
#
# Se crea el objeto Info que almacenará:
# - Dimensiones.
# - Brillo.
# - Ruta de origen.
# - Historial.
#
# Se espera:
# - Que las dimensiones coincidan con la imagen.
#

info = Info(
    dimensiones=data_rx.shape,
    brillo=1.0,
    ruta_origen=ruta
)

print("\nObjeto Info creado correctamente")

# ============================================================
# 5. CREACIÓN DE ImagenRadiografia
# ============================================================
#
# Se instancia la clase ImagenRadiografia utilizando:
# - La imagen.
# - Los metadatos.
# - Tipo de estudio.
# - Condiciones de adquisición.
#
# Se espera:
# - Crear correctamente el objeto.
# - Registrar el evento en historial.
#

radiografia = ImagenRadiografia(
    data=data_rx,
    info=info,
    tipo_estudio="Tórax",
    condiciones_adquisicion="100 kVp, 10 mAs"
)

print("\nObjeto ImagenRadiografia creado")
print(radiografia)

# ============================================================
# 6. VISUALIZACIÓN ORIGINAL
# ============================================================
#
# Se visualiza la radiografía original.
#
# Se espera:
# - Observar correctamente la imagen en escala de grises.
# - Verificar que no existan artefactos visuales.
#

print("\nVisualizando radiografía original")
radiografia.visualizar()

# ============================================================
# 7. SELECCIÓN DE ROI
# ============================================================
#
# La ROI se obtiene sobre la radiografía original antes de
# aplicar transformaciones de contraste, ecualización o
# inversión de intensidad.
#
# Se busca automáticamente una región central con mayor
# variación de intensidades para evitar zonas blancas o negras
# demasiado homogéneas.
#
# Se espera:
# - Obtener una nueva ImagenRadiografia.
# - Mantener historial independiente.
# - Visualizar una zona anatómicamente más útil.
#

print("\nSeleccionando ROI")

x_roi, y_roi, score_roi = buscar_roi_interesante(
    radiografia._data,
    alto=200,
    ancho=200,
    paso=80
)

print(f"ROI sugerida: x={x_roi}, y={y_roi}, score={score_roi:.4f}")

roi = radiografia.seleccionar_roi(
    x=x_roi,
    y=y_roi,
    ancho=200,
    alto=200
)

print(roi)

print("\nEstadísticas ROI")
print("Min:")
print(roi._data.min())
print("Max:")
print(roi._data.max())
print("Media:")
print(roi._data.mean())
print("Desvío:")
print(roi._data.std())

roi.visualizar()

# ============================================================
# 8. AJUSTE DE VISUALIZACIÓN
# ============================================================
#
# Se aplica un ajuste de ventana simple utilizando
# límites de intensidad.
#
# Se espera:
# - Mejorar visualmente determinadas regiones.
# - Comprimir intensidades fuera del rango.
#

print("\nAplicando ajuste de visualización")

radiografia.ajustar_visualizacion(
    vmin=30,
    vmax=220
)

radiografia.visualizar()

# ============================================================
# 9. MEJORA DE CONTRASTE
# ============================================================
#
# Se realiza stretching de contraste.
#
# Se espera:
# - Expandir dinámicamente el histograma.
# - Aumentar diferenciación entre estructuras.
#

print("\nAplicando mejora de contraste")

radiografia.mejorar_contraste()

radiografia.visualizar()

# ============================================================
# 10. INVERSIÓN DE INTENSIDADES
# ============================================================
#
# Se genera un negativo radiográfico.
#
# Se espera:
# - Invertir zonas claras y oscuras.
#

print("\nAplicando inversión de intensidad")

radiografia.invertir_intensidad()

radiografia.visualizar()

# ============================================================
# 11. ECUALIZACIÓN DE HISTOGRAMA
# ============================================================
#
# Se redistribuyen las intensidades utilizando la
# función de distribución acumulada (CDF).
#
# Se espera:
# - Mejorar el contraste global.
# - Obtener una distribución más uniforme.
#

print("\nAplicando ecualización de histograma")

radiografia.ecualizar_histograma()

radiografia.visualizar()

# ============================================================
# 12. DETECCIÓN DE BORDES
# ============================================================
#
# Se aplica el operador Sobel.
#
# Se espera:
# - Resaltar transiciones bruscas de intensidad.
# - Detectar contornos anatómicos.
#

print("\nDetectando bordes")

bordes = radiografia.detectar_bordes()

print("Shape de bordes:")
print(bordes.shape)

plt.figure(figsize=(6, 6))
plt.imshow(bordes, cmap="gray")
plt.title("Detección de Bordes - Sobel")
plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# 13. CLUSTERING DE INTENSIDADES
# ============================================================
#
# Se aplica clustering k-means sobre intensidades.
#
# Se espera:
# - Agrupar regiones similares.
# - Visualizar segmentación aproximada.
#

print("\nAplicando clustering")

radiografia.graficar_clusters(k=3)

# ============================================================
# 14. APLICACIÓN DE FILTRO GAUSSIANO
# ============================================================
#
# Se aplica suavizado gaussiano.
#
# Se espera:
# - Reducir ruido.
# - Suavizar transiciones.
#

print("\nAplicando filtro gaussiano")

filtro_gauss = FiltroGaussiano(
    tamaño=5,
    sigma=1.2
)

radiografia.aplicar_filtro(filtro_gauss)

radiografia.visualizar()

# ============================================================
# 15. APLICACIÓN DE FILTRO DE NITIDEZ
# ============================================================
#
# Se aplica un filtro de realce.
#
# Se espera:
# - Incrementar definición de bordes.
#

print("\nAplicando filtro de nitidez")

filtro_nitidez = FiltroNitidez()

radiografia.aplicar_filtro(filtro_nitidez)

radiografia.visualizar()

# ============================================================
# 16. NORMALIZACIÓN
# ============================================================
#
# Se normaliza la imagen al rango [0, 255].
#
# Se espera:
# - Obtener valores normalizados.
# - Mantener estructura visual.
#

print("\nAplicando normalización")

radiografia.normalizar()

print("Nuevo mínimo:")
print(radiografia._data.min())
print("Nuevo máximo:")
print(radiografia._data.max())

radiografia.visualizar()

# ============================================================
# 17. REGISTRO MANUAL DE TRANSFORMACIÓN
# ============================================================
#
# Se agrega manualmente una descripción al historial.
#
# Se espera:
# - Registrar correctamente el evento.
#

radiografia.registrar_transformacion(
    "Prueba manual completada"
)

# ============================================================
# 18. VISUALIZACIÓN DEL HISTORIAL
# ============================================================
#
# Se imprime el historial completo de operaciones.
#
# Se espera:
# - Ver todas las transformaciones registradas.
#

print("\nHistorial completo:")
print(info["historial"])