import numpy as np
class Conjuntos:

# --------------- FUNCIONES AUXILIARES ----------------
# Funcion para transformar los datos a array
    def datos_as_array(datos):
        distancias_izq, distancias_der, or_x, or_y, ear, umbral_ear, coord_cab = datos
        datos_transformados = np.expand_dims(np.concatenate([distancias_izq, distancias_der, [or_x], [or_y], coord_cab, [ear], [umbral_ear]]), axis=0)
        return datos_transformados


# ------------ CONJUNTO 3 ----------------
# Funcion para transformar el input.txt           
    def conjunto_3(data):
        """
        Entradas: 22
        [0-15] Distancias entre los puntos de referencia de los ojos
                - Medias de los dos ojos
                - Normalizadas entre ellas con min-max
        [16-17] Coordenadas del centro de la cara
        [18-19] Coordenadas de la orientación de la cara
        [20-21] Coordenadas de los puntos de referencia de las cejas       
        """
        # Hace la media de los dos ojos y lo pone como si fuera solo un ojo
        for i in range(0, 16):
            data[:, i] = (data[:, i] + data[:, i + 16]) / 2
        data = np.delete(data, np.s_[16:32], axis=1)

        # Normalizar cada valor de los primeros 16 de cada fila entre ellos mismos
        data[:, :16] = (data[:, :16] - np.min(data[:, :16], axis=1, keepdims=True)) / (np.max(data[:, :16], axis=1, keepdims=True) - np.min(data[:, :16], axis=1, keepdims=True))

        return data
        

