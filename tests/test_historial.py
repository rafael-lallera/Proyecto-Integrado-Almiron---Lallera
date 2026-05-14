##Pruebas de funcionamiento de la clase Historial de forma aislada
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src")
    )
)

from bioimagenes.core.historial import Historial


h = Historial()

print("Historial creado:")
print(h)