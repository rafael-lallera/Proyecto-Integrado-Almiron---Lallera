import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.core.imagen import Imagen
from bioimagenes.core.info import Info 


# -------------------------
# 1. CARGAR IMAGEN
# -------------------------
ruta = "docs\examples\Imagen de prueba RGB.jpg"

img_array = plt.imread(ruta)

# eliminar alfa si existe
if img_array.ndim == 3 and img_array.shape[2] == 4:
    img_array = img_array[:, :, :3]

print("Imagen cargada:", img_array.shape)


# -------------------------
# 2. CREAR OBJETO ORIGINAL
# -------------------------
info = Info(
    dimensiones=img_array.shape,
    brillo=1.0,
    ruta_origen=ruta   # 👈 clave para trazabilidad
)

img = Imagen(img_array, info)

print("\nImagen original:")
print(img)


# -------------------------
# 3. APLICAR RECORTE
# -------------------------
print("\n--- Aplicando recorte ---")

img_recortada = img.recortar(50, 200, 50, 200)

print("\nImagen recortada:")
print(img_recortada)


# -------------------------
# 4. VISUALIZACIÓN
# -------------------------
print("\nMostrando imagen original...")
img.visualizar()

print("\nMostrando imagen recortada...")
img_recortada.visualizar()


# -------------------------
# 5. HISTORIALES
# -------------------------
print("\n--- HISTORIAL ORIGINAL ---")
print(info["historial"])

print("\n--- HISTORIAL RECORTADA ---")
print(img_recortada._info["historial"])


# -------------------------
# 6. VALIDAR TRAZABILIDAD
# -------------------------
print("\n--- VALIDACIÓN TRAZABILIDAD ---")

print("Ruta original en imagen original:")
print(info["ruta_origen"])

print("\nRuta en imagen recortada:")
print(img_recortada._info["ruta_origen"])