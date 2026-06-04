"""
tests/test_manual_radiografia.py

Prueba manual de la clase ImagenRadiografia utilizando
una radiografía real cargada desde disco.
...
"""
import numpy as np
import matplotlib.pyplot as plt
from bioimagenes.core.info import Info
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia
from bioimagenes.filtros.filtro import (
    FiltroGaussiano,
    FiltroNitidez
)


def buscar_roi_interesante(
    data: np.ndarray,
    alto: int = 200,
    ancho: int = 200,
    paso: int = 80
):
    """
    Busca una ROI con mayor variación de intensidades dentro de
    una región central de la imagen.
    """
    filas, columnas = data.shape
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
ruta = r"docs/radiografias/sample/46523715740384360192496023767246369337_veyewt.png"
img = plt.imread(ruta)
print("Radiografía cargada correctamente")
print("Shape original:", img.shape)
print("Tipo de dato original:", img.dtype)

# ============================================================
# 2. CONVERSIÓN A ESCALA DE GRISES
# ============================================================
if img.ndim == 3:
    img = np.mean(img[:, :, :3], axis=2)
print("\nConversión a escala de grises completada")
print("Nuevo shape:", img.shape)

# ============================================================
# 3. CONVERSIÓN A FLOAT64
# ============================================================
data_rx = img.astype(np.float64)
print("\nConversión a float64 realizada")
print("Nuevo dtype:", data_rx.dtype)

# ============================================================
# 4. CREACIÓN DEL OBJETO INFO
# ============================================================
info = Info(
    dimensiones=data_rx.shape,
    brillo=1.0,
    ruta_origen=ruta
)
print("\nObjeto Info creado correctamente")

# ============================================================
# 5. CREACIÓN DE ImagenRadiografia
# ============================================================
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
print("\nVisualizando radiografía original")
radiografia.visualizar()

# ============================================================
# 7. SELECCIÓN DE ROI
# ============================================================
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
print("Min:", roi._data.min())
print("Max:", roi._data.max())
print("Media:", roi._data.mean())
print("Desvío:", roi._data.std())
roi.visualizar()

# ============================================================
# 8. AJUSTE DE VISUALIZACIÓN
# ============================================================
print("\nAplicando ajuste de visualización")
radiografia.ajustar_visualizacion(vmin=30, vmax=220)
radiografia.visualizar()

# ============================================================
# 9. MEJORA DE CONTRASTE
# ============================================================
print("\nAplicando mejora de contraste")
radiografia.mejorar_contraste()
radiografia.visualizar()

# ============================================================
# 10. INVERSIÓN DE INTENSIDADES
# ============================================================
print("\nAplicando inversión de intensidad")
radiografia.invertir_intensidad()
radiografia.visualizar()

# ============================================================
# 11. ECUALIZACIÓN DE HISTOGRAMA
# ============================================================
print("\nAplicando ecualización de histograma")
radiografia.ecualizar_histograma()
radiografia.visualizar()

# ============================================================
# 12. DETECCIÓN DE BORDES
# ============================================================
print("\nDetectando bordes")
bordes = radiografia.detectar_bordes()
print("Shape de bordes:", bordes.shape)
plt.figure(figsize=(6, 6))
plt.imshow(bordes, cmap="gray")
plt.title("Detección de Bordes - Sobel")
plt.axis("off")
plt.tight_layout()
plt.show()


# ============================================================
# 13.3 CLUSTERING DE MÚLTIPLES RADIOGRAFÍAS
# ============================================================
import os

print("\nAplicando clustering de múltiples radiografías")

carpeta = r"docs\radiografias\sample"
extensiones = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

rutas = sorted([
    os.path.join(carpeta, archivo)
    for archivo in os.listdir(carpeta)
    if archivo.lower().endswith(extensiones)
])

print(f"Imágenes encontradas: {len(rutas)}")

lista_radiografias = []
for ruta_img in rutas:
    try:
        raw = plt.imread(ruta_img)
        if raw.ndim == 3:
            raw = np.mean(raw[:, :, :3], axis=2)
        raw = raw.astype(np.float64)
        inf = Info(
            dimensiones=raw.shape,
            brillo=1.0,
            ruta_origen=ruta_img
        )
        lista_radiografias.append(
            ImagenRadiografia(
                data=raw,
                info=inf,
                tipo_estudio="Tórax",
                condiciones_adquisicion="no especificadas"
            )
        )
        print(f"  OK: {os.path.basename(ruta_img)}")
    except Exception as e:
        print(f"  ERROR {os.path.basename(ruta_img)}: {e}")

print(f"\nImágenes cargadas exitosamente: {len(lista_radiografias)}")

if len(lista_radiografias) >= 3:
    ImagenRadiografia.graficar_clusters_imagenes(
        imagenes=lista_radiografias,
        k=3,
        tamaño_thumbnail=64
    )
else:
    print("No hay suficientes imágenes para clustering (mínimo 3)")
# ============================================================
# 14. APLICACIÓN DE FILTRO GAUSSIANO
# ============================================================
print("\nAplicando filtro gaussiano")
filtro_gauss = FiltroGaussiano(tamaño=5, sigma=1.2)
radiografia.aplicar_filtro(filtro_gauss)
radiografia.visualizar()

# ============================================================
# 15. APLICACIÓN DE FILTRO DE NITIDEZ
# ============================================================
print("\nAplicando filtro de nitidez")
filtro_nitidez = FiltroNitidez()
radiografia.aplicar_filtro(filtro_nitidez)
radiografia.visualizar()

# ============================================================
# 16. NORMALIZACIÓN
# ============================================================
print("\nAplicando normalización")
radiografia.normalizar()
print("Nuevo mínimo:", radiografia._data.min())
print("Nuevo máximo:", radiografia._data.max())
radiografia.visualizar()

# ============================================================
# 17. REGISTRO MANUAL DE TRANSFORMACIÓN
# ============================================================
radiografia.registrar_transformacion("Prueba manual completada")

# ============================================================
# 18. VISUALIZACIÓN DEL HISTORIAL
# ============================================================
print("\nHistorial completo:")
print(info["historial"])