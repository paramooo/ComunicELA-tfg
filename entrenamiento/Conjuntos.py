import numpy as np
class Conjuntos:

# --------------- FUNCIONES AUXILIARES ----------------
# Funcion para transformar los datos a array
    def datos_as_array(datos):
        distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, _ = datos
        datos_transformados = np.expand_dims(np.concatenate([distancias_izq, distancias_der, [or_x], [or_y], [or_z], coord_cab, [ear], [umbral_ear]]), axis=0)
        return datos_transformados


#----------------CONJUNTOS PARA EL INPUT 0 ----------------
# ------------ CONJUNTO 1 ----------------
# Funcion para transformar el input.txt
    def conjunto_1(data):
        """
        Entradas: 39 -> TODOS LOS DATOS (NORMALIZAR MIN MAX DISTANCIAS)
        [0-15] Distancias entre los puntos de referencia ojo derecho min max
        [16-31] Distancias entre los puntos de referencia ojo izquierdo min max
        [32-34] Coordenadas de la orientación de la cara
        [35-36] Coordenadas del centro de la cara
        [37-38] EAR y umbral EAR 
        """
        # Normalizar cada valor de los primeros 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO DERECHO
        data[:, :16] = (data[:, :16] - np.min(data[:, :16], axis=1, keepdims=True)) / (np.max(data[:, :16], axis=1, keepdims=True) - np.min(data[:, :16], axis=1, keepdims=True))

        # Normalizar cada valor de los segundos 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO IZQUIERDO
        data[:, 16:32] = (data[:, 16:32] - np.min(data[:, 16:32], axis=1, keepdims=True)) / (np.max(data[:, 16:32], axis=1, keepdims=True) - np.min(data[:, 16:32], axis=1, keepdims=True))

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

        return data
    



        
    def conjunto_3(data):
        """
        Entradas: 39
        [0-31] Distancias entre los puntos de referencia de los ojos (min-max)
        [32-34] Coordenadas de la orientación de la cara reducido el rango a 0.3-0.7
        [35-36] Coordenadas del centro de la cara reducido el rango a 0.3-0.7
        [37-38] EAR y umbral EAR
        """
        # Normalizar cada valor de los primeros 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO DERECHO
        data[:, :16] = (data[:, :16] - np.min(data[:, :16], axis=1, keepdims=True)) / (np.max(data[:, :16], axis=1, keepdims=True) - np.min(data[:, :16], axis=1, keepdims=True))

        # Normalizar cada valor de los segundos 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO IZQUIERDO
        data[:, 16:32] = (data[:, 16:32] - np.min(data[:, 16:32], axis=1, keepdims=True)) / (np.max(data[:, 16:32], axis=1, keepdims=True) - np.min(data[:, 16:32], axis=1, keepdims=True))

        # Pasar las cifras de entre 0.3 y 0.7 a 0 y 1
        data[:, 32:37] = (data[:, 32:37] - 0.3) / (0.7 - 0.3)
        
        # Todo clip entre 0 y 1
        data[:, :] = np.clip(data[:, :], 0, 1)

        return data



    def conjunto_4(data):
        """
            Entradas: 37
            [0-31] Distancias entre los puntos de referencia de los ojos(min-max)
            [32-34] Coordenadas de la orientación de la cara
            [35-36] Coordenadas del centro de la cara
        """
        # Normalizar cada valor de los primeros 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO DERECHO
        data[:, :16] = (data[:, :16] - np.min(data[:, :16], axis=1, keepdims=True)) / (np.max(data[:, :16], axis=1, keepdims=True) - np.min(data[:, :16], axis=1, keepdims=True))

        # Normalizar cada valor de los segundos 16 de cada fila entre ellos mismos PARA NORMALIZAR EL OJO IZQUIERDO
        data[:, 16:32] = (data[:, 16:32] - np.min(data[:, 16:32], axis=1, keepdims=True)) / (np.max(data[:, 16:32], axis=1, keepdims=True) - np.min(data[:, 16:32], axis=1, keepdims=True))

        # Todo clip entre 0 y 1
        data[:, :] = np.clip(data[:, :], 0, 1)
        
        # Elimina las columnas 37-38 de todas las filas
        data = np.delete(data, np.s_[37:39], axis=1)
        return data