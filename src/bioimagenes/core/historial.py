class Historial:
    """
    Clase que almacena y gestiona los cambios realizados sobre una imagen.
    """

    # -------------------------
    # CONSTRUCTOR
    # -------------------------
    def __init__(self):
        """
        Se ejecuta cuando se crea un objeto Historial.
        Inicializa una lista vacía donde se guardarán los cambios.
        """

        # Lista privada donde se guardan los cambios
        # Ejemplo: ["Filtro aplicado", "Recorte", "Brillo ajustado"]
        self._lista_cambios = []

    # -------------------------
    # AGREGAR CAMBIO
    # -------------------------
    def modificar_historial(self, cambio: str):
        """
        Agrega un nuevo cambio al historial.

        Parámetro:
        - cambio: texto que describe la modificación realizada
        """

        # Validamos que el cambio sea texto
        if not isinstance(cambio, str):
            raise TypeError("El cambio debe ser un string")

        # Agregamos el cambio a la lista
        self._lista_cambios.append(cambio)

    # -------------------------
    # ÚLTIMO CAMBIO
    # -------------------------
    @property
    def ultimo_cambio(self):
        """
        Devuelve el último cambio realizado.
        Si no hay cambios, devuelve None.
        """

        if len(self._lista_cambios) == 0:
            return None

        return self._lista_cambios[-1]

    # -------------------------
    # LONGITUD DEL HISTORIAL
    # -------------------------
    def __len__(self):
        """
        Permite usar len(historial)
        Devuelve la cantidad de cambios registrados.
        """

        return len(self._lista_cambios)

    # -------------------------
    # ITERAR SOBRE LOS CAMBIOS
    # -------------------------
    def __iter__(self):
        """
        Permite recorrer el historial con un for.
        """

        return iter(self._lista_cambios)

    # -------------------------
    # REPRESENTACIÓN EN TEXTO
    # -------------------------
    def __str__(self):
        """
        Define cómo se muestra el historial al imprimirlo.
        """

        # Si no hay cambios
        if not self._lista_cambios:
            return "Sin cambios"

        # Une los cambios en líneas separadas
        return "\n".join(self._lista_cambios)