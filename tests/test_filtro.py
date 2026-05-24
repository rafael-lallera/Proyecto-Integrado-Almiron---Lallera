import numpy as np
import matplotlib.pyplot as plt

from bioimagenes.filtros.filtro import (
    FiltroGaussiano,
    FiltroMediana,
    FiltroSuavizado,
    FiltroNitidez
)


# -------------------------
# 1. CARGAR IMAGEN
# -------------------------
ruta = "docs/examples/Imagen de prueba RGB.jpg"

img_array = plt.imread(ruta)

# eliminar canal alfa si existe
if img_array.ndim == 3 and img_array.shape[2] == 4:
    img_array = img_array[:, :, :3]

print("Imagen cargada correctamente")
print("Shape:", img_array.shape)
print("Tipo:", img_array.dtype)


# -------------------------
# 2. CREAR FILTROS
# -------------------------
filtro_gauss = FiltroGaussiano(
    tamaño=5,
    sigma=1.5
)

filtro_mediana = FiltroMediana(
    tamaño=3
)

filtro_suavizado = FiltroSuavizado(
    tamaño=5
)

filtro_nitidez = FiltroNitidez()


# -------------------------
# 3. MOSTRAR INFO FILTROS
# -------------------------
print("\n--- FILTROS CREADOS ---")

print("\nFiltro Gaussiano:")
print(filtro_gauss)
print(filtro_gauss.kernel)

print("\nFiltro Mediana:")
print(filtro_mediana)

print("\nFiltro Suavizado:")
print(filtro_suavizado)
print(filtro_suavizado.kernel)

print("\nFiltro Nitidez:")
print(filtro_nitidez)
print(filtro_nitidez.kernel)


# -------------------------
# 4. APLICAR FILTROS
# -------------------------
print("\n--- APLICANDO FILTROS ---")

img_gauss = filtro_gauss.aplicar_filtro(img_array)
print("Filtro gaussiano aplicado")

img_mediana = filtro_mediana.aplicar_filtro(img_array)
print("Filtro mediana aplicado")

img_suavizada = filtro_suavizado.aplicar_filtro(img_array)
print("Filtro suavizado aplicado")

img_nitidez = filtro_nitidez.aplicar_filtro(img_array)
print("Filtro nitidez aplicado")


# -------------------------
# 5. VISUALIZACIÓN
# -------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# ORIGINAL
ax[0, 0].imshow(img_array)
ax[0, 0].set_title("Original")
ax[0, 0].axis("off")

# GAUSSIANO
ax[0, 1].imshow(img_gauss)
ax[0, 1].set_title("Gaussiano")
ax[0, 1].axis("off")

# MEDIANA
ax[0, 2].imshow(img_mediana)
ax[0, 2].set_title("Mediana")
ax[0, 2].axis("off")

# SUAVIZADO
ax[1, 0].imshow(img_suavizada)
ax[1, 0].set_title("Suavizado")
ax[1, 0].axis("off")

# NITIDEZ
ax[1, 1].imshow(img_nitidez)
ax[1, 1].set_title("Nitidez")
ax[1, 1].axis("off")

# espacio vacío
ax[1, 2].axis("off")

plt.tight_layout()
plt.show()


# -------------------------
# 6. COMPARACIÓN NUMÉRICA
# -------------------------
print("\n--- INFORMACIÓN NUMÉRICA ---")

print("\nOriginal:")
print("Min:", np.min(img_array))
print("Max:", np.max(img_array))
print("Mean:", np.mean(img_array))

print("\nGaussiano:")
print("Min:", np.min(img_gauss))
print("Max:", np.max(img_gauss))
print("Mean:", np.mean(img_gauss))

print("\nMediana:")
print("Min:", np.min(img_mediana))
print("Max:", np.max(img_mediana))
print("Mean:", np.mean(img_mediana))

print("\nSuavizado:")
print("Min:", np.min(img_suavizada))
print("Max:", np.max(img_suavizada))
print("Mean:", np.mean(img_suavizada))

print("\nNitidez:")
print("Min:", np.min(img_nitidez))
print("Max:", np.max(img_nitidez))
print("Mean:", np.mean(img_nitidez))


# -------------------------
# 7. VALIDAR SHAPES
# -------------------------
print("\n--- VALIDACIÓN SHAPES ---")

print("Original:", img_array.shape)
print("Gaussiano:", img_gauss.shape)
print("Mediana:", img_mediana.shape)
print("Suavizado:", img_suavizada.shape)
print("Nitidez:", img_nitidez.shape)


# -------------------------
# 8. VALIDAR TIPOS
# -------------------------
print("\n--- VALIDACIÓN DTYPES ---")

print("Original:", img_array.dtype)
print("Gaussiano:", img_gauss.dtype)
print("Mediana:", img_mediana.dtype)
print("Suavizado:", img_suavizada.dtype)
print("Nitidez:", img_nitidez.dtype)