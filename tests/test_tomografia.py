"""
test_tomografia.py

Test manual de la clase ImagenTomografia.

Permite verificar:
- Carga de un archivo NIfTI (.nii / .nii.gz)
- Resumen general del volumen
- Visualización de cortes axial, coronal y sagital
- Ajuste de ventana y presets de tejido
- Segmentación de un corte por tejidos
- Reconstrucción 3D simplificada (MIP)
- Exploración interactiva de slices con Plotly
- Normalización de intensidades
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================================================
# Ajuste de ruta para importar el paquete desde src/
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from bioimagenes.medicas.imagen_tomografia import ImagenTomografia


# ==========================================================
# RUTA DEL ARCHIVO NIfTI
# ==========================================================
ARCHIVO_NIFTI = os.path.join(
    ROOT,
    "docs",
    "tomografia",
    "AC421363f.nii",
    "AC421363f.nii"
)


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def cargar_tomografia():
    """
    Carga la tomografía desde el archivo NIfTI del proyecto.
    """
    if not os.path.exists(ARCHIVO_NIFTI):
        raise FileNotFoundError(
            f"No se encontró el archivo NIfTI en: {ARCHIVO_NIFTI}"
        )

    tc = ImagenTomografia.desde_nifti(
        ARCHIVO_NIFTI,
        window_center=40.0,
        window_width=400.0
    )
    return tc


def mostrar_resumen(tc):
    """
    Imprime un resumen del objeto tomográfico.
    """
    print("=" * 70)
    print("RESUMEN DE LA TOMOGRAFÍA")
    print("=" * 70)
    print(tc)
    print(f"Shape del volumen: {tc._data.shape}")
    print(f"Cantidad de slices: {tc.n_slices}")
    print(f"Voxel spacing: {tc.voxel_spacing}")
    print(f"Origen: {tc.origen}")
    print(f"Escala de intensidades: {tc.escala_intensidad}")
    print(f"Window center: {tc.window_center}")
    print(f"Window width: {tc.window_width}")
    print(f"Dimensiones físicas: {tc.dimensiones_fisicas()}")
    print()


def aplicar_ventana_volumen(volumen, center, width):
    """
    Aplica windowing a todo el volumen y lo lleva a [0, 255].
    """
    hu_min = center - width / 2.0
    hu_max = center + width / 2.0

    vol = np.clip(volumen, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min) * 255.0
    return vol.astype(np.uint8)


def mostrar_volumen_interactivo(volumen, titulo):
    """
    Muestra un volumen 3D con slider usando Plotly.
    El eje 0 se usa como frames de animación.
    """
    fig = px.imshow(
        volumen,
        animation_frame=0,
        color_continuous_scale="gray",
        origin="lower",
        labels=dict(
            animation_frame="Corte",
            color="Intensidad"
        ),
    )
    fig.update_layout(
        title=titulo,
        width=800,
        height=800
    )
    fig.show()


def mostrar_tres_planos_plotly(tc):
    """
    Muestra el volumen completo en los tres planos
    usando visores interactivos con Plotly.
    """
    vol = tc._data
    vol_ventaneado = aplicar_ventana_volumen(
        vol,
        tc.window_center,
        tc.window_width
    )

    # Plano axial: ya está en formato (cortes, filas, columnas)
    mostrar_volumen_interactivo(
        vol_ventaneado,
        "Tomografía - Plano Axial"
    )

    # Plano coronal: el eje de corte pasa a ser el primero
    vol_coronal = np.transpose(vol_ventaneado, (1, 0, 2))
    mostrar_volumen_interactivo(
        vol_coronal,
        "Tomografía - Plano Coronal"
    )

    # Plano sagital: el eje de corte pasa a ser el primero
    vol_sagital = np.transpose(vol_ventaneado, (2, 0, 1))
    mostrar_volumen_interactivo(
        vol_sagital,
        "Tomografía - Plano Sagital"
    )


# ==========================================================
# MAIN
# ==========================================================
def main():
    # ------------------------------------------------------
    # CARGA
    # ------------------------------------------------------
    tc = cargar_tomografia()
    mostrar_resumen(tc)

    # Índices centrales para cada plano
    axial_idx = tc._data.shape[0] // 2
    coronal_idx = tc._data.shape[1] // 2
    sagital_idx = tc._data.shape[2] // 2

    # ------------------------------------------------------
    # VISUALIZACIÓN BÁSICA DE CORTES
    # ------------------------------------------------------
    print("Mostrando corte axial central...")
    tc.mostrar_slice(indice=axial_idx, eje="axial")

    print("Mostrando corte coronal central...")
    tc.mostrar_slice(indice=coronal_idx, eje="coronal")

    print("Mostrando corte sagital central...")
    tc.mostrar_slice(indice=sagital_idx, eje="sagital")

    # ------------------------------------------------------
    # VISUALIZACIÓN GENERAL
    # ------------------------------------------------------
    print("Mostrando visualización general...")
    tc.visualizar()

    # ------------------------------------------------------
    # AJUSTE DE VENTANA
    # ------------------------------------------------------
    print("Ajustando ventana manualmente...")
    tc_ventana = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_ventana.ajustar_ventana(center=50, width=350)
    tc_ventana.mostrar_slice(indice=axial_idx, eje="axial")

    # ------------------------------------------------------
    # PRESETS DE TEJIDO
    # ------------------------------------------------------
    print("Aplicando preset pulmonar...")
    tc_pulmon = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_pulmon.aplicar_preset("pulmon")
    tc_pulmon.mostrar_slice(indice=axial_idx, eje="axial")

    print("Aplicando preset hueso...")
    tc_hueso = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_hueso.aplicar_preset("hueso")
    tc_hueso.mostrar_slice(indice=axial_idx, eje="axial")

    print("Aplicando preset tejido blando...")
    tc_blando = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_blando.aplicar_preset("tejido_blando")
    tc_blando.mostrar_slice(indice=axial_idx, eje="axial")

    # ------------------------------------------------------
    # SEGMENTACIÓN POR TEJIDOS
    # ------------------------------------------------------
    print("Mostrando segmentación por tejidos...")
    tc_segmentacion = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_segmentacion.visualizar_corte(
        indice=axial_idx,
        eje="axial",
        mostrar_original=True
    )

    print("Mostrando segmentación por tejidos sin original...")
    tc_segmentacion.visualizar_corte(
        indice=axial_idx,
        eje="axial",
        mostrar_original=False
    )

    # ------------------------------------------------------
    # RECONSTRUCCIÓN 3D SIMPLIFICADA (MIP)
    # ------------------------------------------------------
    print("Mostrando reconstrucción 3D simplificada (MIP axial)...")
    tc_mip = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)
    tc_mip.reconstruir_3d(eje=0)

    print("Mostrando reconstrucción 3D simplificada (MIP coronal)...")
    tc_mip.reconstruir_3d(eje=1)

    # ------------------------------------------------------
    # NORMALIZACIÓN DE INTENSIDADES
    # ------------------------------------------------------
    print("Probando normalización de intensidades...")
    tc_norm = ImagenTomografia.desde_nifti(ARCHIVO_NIFTI)

    corte_original = tc_norm.obtener_corte(axial_idx, "axial")
    print(
        f"Rango antes de normalizar: "
        f"{corte_original.min():.2f} a {corte_original.max():.2f}"
    )

    tc_norm.normalizar_intensidades(hu_min=-1000, hu_max=1000)

    corte_norm = tc_norm.obtener_corte(axial_idx, "axial")
    print(
        f"Rango después de normalizar: "
        f"{corte_norm.min():.2f} a {corte_norm.max():.2f}"
    )

    # ------------------------------------------------------
    # REGISTRO MANUAL DE TRANSFORMACIONES
    # ------------------------------------------------------
    print("Registrando una transformación manual en el historial...")
    tc.registrar_transformacion(
        "Test manual de la clase ImagenTomografia"
    )

    # ------------------------------------------------------
    # VISORES INTERACTIVOS CON PLOTLY
    # ------------------------------------------------------
    print("Abriendo visor interactivo axial...")
    print("Abriendo visor interactivo coronal...")
    print("Abriendo visor interactivo sagital...")
    mostrar_tres_planos_plotly(tc)

    print("\nFin del test de tomografía.")


if __name__ == "__main__":
    main()