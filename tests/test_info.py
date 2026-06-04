"""
tests/test_manual_info.py

Prueba manual de la clase Info.

Descripción general
-------------------
Este script verifica el funcionamiento de la clase Info
de forma aislada, incluyendo:

- Creación del objeto.
- Acceso a atributos mediante indexación.
- Uso del operador "in".
- Interacción con el historial asociado.
- Iteración sobre el historial.
- Manejo de errores.
- Validación de parámetros de inicialización.

Dependencias
------------
- bioimagenes.core.info

Autor
-----
Proyecto Bioimágenes
"""

from bioimagenes.core.info import Info


# ============================================================
# 1. CREACIÓN DEL OBJETO INFO
# ============================================================
#
# Se crea una instancia básica de Info.
#
# Se espera:
# - Que el objeto se construya correctamente.
# - Que los valores iniciales sean válidos.
#

info = Info(
    dimensiones=(10, 10),
    brillo=0.5
)

print("Objeto Info creado correctamente")

print("\nRepresentación del objeto:")
print(info)

# ============================================================
# 2. ACCESO A ATRIBUTOS
# ============================================================
#
# Se verifica el acceso mediante __getitem__().
#
# Se espera:
# - Recuperar correctamente los atributos almacenados.
#

print("\nAcceso a atributos")

print("Dimensiones:", info["dimensiones"])
print("Brillo:", info["brillo"])
print("Cortada:", info["cortada"])

# ============================================================
# 3. OPERADOR CONTAINS
# ============================================================
#
# Se verifica el funcionamiento del operador "in".
#
# Se espera:
# - Detectar claves válidas.
# - Rechazar claves inexistentes.
#

print("\nPrueba del operador 'in'")

print("¿Existe 'brillo'?", "brillo" in info)
print("¿Existe 'historial'?", "historial" in info)
print("¿Existe 'invalido'?", "invalido" in info)

# ============================================================
# 4. PRUEBA DEL HISTORIAL
# ============================================================
#
# Se agregan eventos al historial asociado.
#
# Se espera:
# - Registrar correctamente los cambios.
# - Mantener el orden de inserción.
#

print("\nProbando historial")

historial = info["historial"]

historial.modificar_historial(
    "Filtro aplicado"
)

historial.modificar_historial(
    "Recorte realizado"
)

print("\nContenido del historial:")
print(historial)

# ============================================================
# 5. ÚLTIMO CAMBIO
# ============================================================
#
# Se verifica la propiedad ultimo_cambio.
#
# Se espera:
# - Recuperar el evento más reciente.
#

print("\nÚltimo cambio registrado:")
print(historial.ultimo_cambio)

# ============================================================
# 6. CANTIDAD DE CAMBIOS
# ============================================================
#
# Se verifica el método __len__().
#
# Se espera:
# - Obtener la cantidad correcta de registros.
#

print("\nCantidad de cambios:")
print(len(historial))

# ============================================================
# 7. ITERACIÓN DEL HISTORIAL
# ============================================================
#
# Se recorre el historial utilizando un bucle for.
#
# Se espera:
# - Acceder a cada evento en orden cronológico.
#

print("\nIterando historial")

for cambio in historial:
    print("-", cambio)

# ============================================================
# 8. ACCESO A CLAVE INVÁLIDA
# ============================================================
#
# Se intenta acceder a una clave inexistente.
#
# Se espera:
# - Capturar la excepción correspondiente.
#

print("\nProbando clave inválida")

try:
    print(info["no_existe"])

except Exception as error:
    print("Error capturado:")
    print(error)

# ============================================================
# 9. VALIDACIÓN DE ERRORES DE INICIALIZACIÓN
# ============================================================
#
# Se prueban distintos escenarios inválidos.
#
# Se espera:
# - Que la clase rechace entradas incorrectas.
#

print("\nProbando errores de inicialización")

# ------------------------------------------------------------
# dimensiones con tipo incorrecto
# ------------------------------------------------------------

try:
    Info(
        dimensiones=[10, 10],
        brillo=0.5
    )

except Exception as error:
    print("\nError dimensiones:")
    print(error)

# ------------------------------------------------------------
# dimensión negativa
# ------------------------------------------------------------

try:
    Info(
        dimensiones=(10, -5),
        brillo=0.5
    )

except Exception as error:
    print("\nError dimensión negativa:")
    print(error)

# ------------------------------------------------------------
# brillo inválido
# ------------------------------------------------------------

try:
    Info(
        dimensiones=(10, 10),
        brillo="alto"
    )

except Exception as error:
    print("\nError brillo:")
    print(error)

# ------------------------------------------------------------
# historial inválido
# ------------------------------------------------------------

try:
    Info(
        dimensiones=(10, 10),
        brillo=0.5,
        historial="cualquiera"
    )

except Exception as error:
    print("\nError historial:")
    print(error)

# ============================================================
# 10. PRUEBA COMPLETA DE FUNCIONAMIENTO
# ============================================================
#
# Se crea una nueva instancia y se registran eventos
# en el historial.
#
# Se espera:
# - Verificar la interacción conjunta de Info e Historial.
#

print("\nPRUEBA COMPLETA")

info = Info(
    dimensiones=(5, 5),
    brillo=1.0
)

info["historial"].modificar_historial(
    "Inicialización"
)

info["historial"].modificar_historial(
    "Ajuste de brillo"
)

print("\nObjeto Info:")
print(info)

print("\nHistorial asociado:")
print(info["historial"])

print("\nÚltimo cambio registrado:")
print(info["historial"].ultimo_cambio)