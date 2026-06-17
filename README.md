# bioimagenes

Librería en Python para el procesamiento, análisis y visualización de imágenes biomédicas. El proyecto está organizado con un layout `src` y separa la lógica común de metadatos e historial de las clases especializadas para radiografías, termografías y tomografías. Además incluye una colección de filtros espaciales y utilidades de visualización pensadas para trabajar con datos reales y para conservar trazabilidad de las transformaciones aplicadas sobre cada imagen.

## Qué incluye el proyecto

El núcleo del paquete está formado por `Info`, que administra metadatos, y `Historial`, que registra transformaciones y eventos sobre cada imagen. Sobre esa base se construye `Imagen`, la clase general para matrices de imagen, con operaciones como conversión a escala de grises, normalización, recorte, indexación, aplicación de filtros y visualización.

Encima de esa capa común aparecen las clases médicas específicas. `ImagenRadiografia` agrega operaciones de realce, inversión de intensidad, ecualización de histograma, detección de bordes, selección de ROI y una visualización orientada a imágenes radiográficas. `ImagenTermografica` incorpora interpretación térmica, conversión entre unidades, detección de puntos calientes, segmentación por umbral y mapa de calor. `ImagenTomografia` trabaja con volúmenes 3-D, carga desde NIfTI, windowing, presets de tejido, reconstrucción tipo MIP y exploración interactiva de cortes.

El módulo `filtros` reúne implementaciones reutilizables como `FiltroGaussiano`, `FiltroMediana`, `FiltroSuavizado` y `FiltroNitidez`. El proyecto también incluye datos de ejemplo en `docs/` y scripts de prueba manual en `tests/`.

## Estructura principal

```text
src/bioimagenes/
├── core/            # Info, Historial e Imagen base
├── filtros/         # Filtros espaciales
├── medicas/         # Radiografía, termografía y tomografía
├── visualizacion/   # Espacio para utilidades de visualización
└── version.py

docs/
├── examples/
├── radiografias/
├── termografias/
└── tomografia/

tests/
└── test_*.py
```

## Requisitos

El paquete requiere Python 3.11 o superior. En tiempo de ejecución usa `numpy`, `matplotlib`, `scipy`, `pillow`, `nibabel` y `plotly`.

## Instalación

### Con pip

```bash
git clone <url-del-repositorio>
cd Proyecto_integrador
pip install -e .
```

Para instalar también las dependencias de desarrollo y pruebas:

```bash
pip install -e ".[dev]"
```

### Con conda

El repositorio incluye un entorno reproducible en `ambiente.yml`:

```bash
conda env create -f ambiente.yml
conda activate miEntornoPDA
```

## Uso rápido

### Crear una imagen base

```python
import matplotlib.pyplot as plt
from bioimagenes.core.info import Info
from bioimagenes.core.imagen import Imagen

data = plt.imread("docs/examples/Imagen de prueba RGB.jpg")
if data.ndim == 3 and data.shape[2] == 4:
    data = data[:, :, :3]

info = Info(dimensiones=data.shape, brillo=1.0, ruta_origen="docs/examples/Imagen de prueba RGB.jpg")
img = Imagen(data, info)

img_bn = img.bn()
img_recortada = img.recortar(50, 200, 50, 200)
```

### Trabajar con radiografías

```python
import numpy as np
import matplotlib.pyplot as plt
from bioimagenes.core.info import Info
from bioimagenes.medicas.imagen_radiografia import ImagenRadiografia

ruta = "docs/radiografias/sample/46523715740384360192496023767246369337_veyewt.png"
img = plt.imread(ruta)
if img.ndim == 3:
    img = np.mean(img[:, :, :3], axis=2)

info = Info(dimensiones=img.shape, brillo=1.0, ruta_origen=ruta)
rx = ImagenRadiografia(img.astype(np.float64), info, tipo_estudio="Tórax", condiciones_adquisicion="100 kVp, 10 mAs")
rx.mejorar_contraste()
roi = rx.seleccionar_roi(x=100, y=100, ancho=200, alto=200)
```

### Trabajar con termografías

```python
import numpy as np
import matplotlib.pyplot as plt
from bioimagenes.core.info import Info
from bioimagenes.medicas.imagen_termografica import ImagenTermografica

ruta = "docs/termografias/mate.jpg"
img = plt.imread(ruta)
if img.ndim == 3:
    img = np.mean(img[:, :, :3], axis=2)

info = Info(dimensiones=img.shape, brillo=1.0, ruta_origen=ruta)
termo = ImagenTermografica(
    img.astype(np.float64),
    info,
    unidad="C",
    rango_temp=(float(img.min()), float(img.max())),
    escala=(1.0, 0.0),
    umbral_calor=float(np.percentile(img, 75)),
)
puntos_calientes = termo.detectar_puntos_calientes()
```

### Trabajar con tomografías

```python
from bioimagenes.medicas.imagen_tomografia import ImagenTomografia

tc = ImagenTomografia.desde_nifti("docs/tomografia/AC421363f.nii/AC421363f.nii")
tc.aplicar_preset("tejido_blando")
corte = tc.obtener_corte(indice=tc.n_slices // 2, eje="axial")
```

## Pruebas

Los archivos de `tests/` son scripts de validación manual y ejemplos de uso sobre los datos incluidos en `docs/`. Pueden ejecutarse directamente o adaptarse a una batería de pruebas automatizadas con `pytest`.

## Autoría y licencia

Autores: Fernando Almirón y Rafael Lallera.  
Licencia: MIT.
