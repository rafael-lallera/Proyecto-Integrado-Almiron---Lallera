##Pruebas de funcionamiento de la clase info de forma aislada
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src")
    )
)

from bioimagenes.core.info import Info

info = Info((10, 10), 0.5)

print("Objeto Info creado:")
print(info)

print("\nAcceso a atributos:")

print("Dimensiones:", info["dimensiones"])
print("Brillo:", info["brillo"])
print("Cortada:", info["cortada"])

print("\nPrueba de contains:")

print("¿Existe 'brillo'?", "brillo" in info)
print("¿Existe 'historial'?", "historial" in info)
print("¿Existe 'invalido'?", "invalido" in info)

print("\nProbando historial:")

hist = info["historial"]

hist.modificar_historial("Filtro aplicado")
hist.modificar_historial("Recorte")

print(hist)

print("\nÚltimo cambio:")
print(hist.ultimo_cambio)

print("\nCantidad de cambios:")
print(len(hist))

print("\nIterando historial:")

for cambio in hist:
    print("-", cambio)

print("\nProbando clave inválida:")

try:
    print(info["no_existe"])
except Exception as e:
    print("Error capturado:", e)

print("\nProbando errores de inicialización:")

# dimensiones mal tipo
try:
    Info([10, 10], 0.5)
except Exception as e:
    print("Error dimensiones:", e)

# dimensión negativa
try:
    Info((10, -5), 0.5)
except Exception as e:
    print("Error dimensión negativa:", e)

# brillo incorrecto
try:
    Info((10, 10), "alto")
except Exception as e:
    print("Error brillo:", e)

# historial inválido
try:
    Info((10, 10), 0.5, historial="cualquiera")
except Exception as e:
    print("Error historial:", e)

print("\nPRUEBA COMPLETA:")

info = Info((5, 5), 1.0)

info["historial"].modificar_historial("Inicialización")
info["historial"].modificar_historial("Ajuste de brillo")

print("\nInfo:")
print(info)

print("\nHistorial:")
print(info["historial"])

print("\nÚltimo cambio:")
print(info["historial"].ultimo_cambio)