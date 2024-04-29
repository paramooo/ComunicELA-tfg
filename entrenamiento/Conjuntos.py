import numpy as np
class Conjuntos:

# --------------- FUNCIONES AUXILIARES ----------------
# Funcion para transformar los datos a array
    def datos_as_array(datos):
        distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab = datos
        datos_transformados = np.expand_dims(np.concatenate([distancias_izq, distancias_der, [or_x], [or_y], [or_z], coord_cab, [ear], [umbral_ear]]), axis=0)
        return datos_transformados


#----------------CONJUNTOS PARA EL INPUT 0 ----------------
# ------------ CONJUNTO 1 ----------------
# Funcion para transformar el input.txt
    def conjunto_1(data):
        """
        Entradas: 14
        [0-5] Distancias entre los puntos de referencia del ojo izquierdo 
        [6-11] Distancias entre los puntos de referencia del ojo derecho
        [19] EAR
        [20] Umbral EAR
        MAAAAAAAAAAAAAAAAL
        """
        # Coje los puntos 0,2,4,6,8,10 (ojo izquierdo) y 1,3,5,7,9,11 (ojo derecho) y el EAR y el umbral EAR
        data = np.concatenate((data[:, 0:12:2], data[:, 1:12:2], data[:, 19:21]), axis=1)

        return data

# ------------ CONJUNTO 2 ----------------
# Funcion para transformar el input.txt           
    def conjunto_2(data):
        """
        Entradas: 23
        [0-15] Distancias entre los puntos de referencia de los ojos
                - Medias de los dos ojos
                - Normalizadas entre ellas con min-max
        [16-18] Coordenadas de la orientación de la cara
        [19-20] Coordenadas del centro de la cara
        [21-22] EAR y umbral EAR      
        """
        # Hace la media de los dos ojos y lo pone como si fuera solo un ojo
        for i in range(0, 16):
            data[:, i] = (data[:, i] + data[:, i + 16]) / 2
        data = np.delete(data, np.s_[16:32], axis=1)

        # Normalizar cada valor de los primeros 16 de cada fila entre ellos mismos
        data[:, :16] = (data[:, :16] - np.min(data[:, :16], axis=1, keepdims=True)) / (np.max(data[:, :16], axis=1, keepdims=True) - np.min(data[:, :16], axis=1, keepdims=True))

        # Pasar las cifras de entre 0.3 y 0.7 a 0 y 1
        data[:, 16:21] = (data[:, 16:21] - 0.3) / (0.7 - 0.3)
        # A 0 y 1
        data[:, 16:21] = np.clip(data[:, 16:21], 0, 1)

        return data
    
        
