"""
tests/test_historial.py

Prueba manual de la clase Historial.

Descripción general
-------------------
Este script verifica el funcionamiento básico de la clase
Historial de forma aislada.

El objetivo es comprobar:

- La creación correcta del objeto.
- La inicialización del historial interno.
- La representación textual mediante print().

Dependencias
------------
- bioimagenes.core.historial

Autor
-----
Proyecto Bioimágenes
"""

from bioimagenes.core.historial import Historial


# ============================================================
# 1. CREACIÓN DEL HISTORIAL
# ============================================================
#
# Se instancia un objeto Historial vacío.
#
# Se espera:
# - Que la creación no genere excepciones.
# - Que el historial se inicialice correctamente.
#

historial = Historial()

print("Historial creado correctamente")

# ============================================================
# 2. VISUALIZACIÓN DEL CONTENIDO
# ============================================================
#
# Se imprime la representación textual del historial.
#
# Se espera:
# - Observar el estado inicial del historial.
# - Verificar el funcionamiento del método __str__().
#

print("\nContenido del historial:")

print(historial)