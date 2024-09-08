from mediapipe.python.solutions.face_mesh import FaceMesh
from numpy import linalg as np_linalg, array as np_array, zeros as np_zeros, clip as np_clip
from cv2 import solvePnP as cv2_solvePnP, Rodrigues as cv2_Rodrigues, decomposeProjectionMatrix as cv2_decomposeProjectionMatrix, \
    hconcat as cv2_hconcat, SOLVEPNP_ITERATIVE as cv2_SOLVEPNP_ITERATIVE

class Detector:
    """ 
    Clase que contiene las funciones de detección de los puntos de referencia de la cara y el cálculo del EAR
    """

    def __init__(self):
        # Índices de los puntos de referencia para los ojos, primero las pupilas y despues el resto
        self.indices_ojo_izq = [473, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]  
        self.indices_ojo_der = [468, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.indices_ear_der = [133, 158, 160, 33, 144, 153]
        self.indices_ear_izq = [362, 385, 387, 263, 373, 380]
        self.incide_central = [8]
        self.indices_orientacion = [4, 152, 263, 33, 287, 57]

        #Modelo de MediaPipe
        self.deteccion_cara = FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        
        
    def obtener_coordenadas_indices(self, frame):
        """
        Función que obtiene las coordenadas de los puntos de referencia de la cara en un frame

        Args:
            frame (np.array): Frame a procesar  
        
        Returns:
            tuple: Coordenadas de los puntos de referencia de los ojos izquierdo y derecho,
                   coordenadas de los puntos de referencia de los ojos para el ear,
                   coordenadas del centro de la cara, 
                   coordenadas de los puntos necesarios para el calculo de la orientación de la cara

        """
        if frame is None:
            return None
        altura, ancho, _ = frame.shape              
        resultados = self.deteccion_cara.process(frame)  

        # Init de las listas de coordenadas
        coordenadas_ojo_izq = []
        coordenadas_ojo_der = [] 
        coordenadas_ear_izq = []
        coordenadas_ear_der = []
        coordenadas_central = None
        coordenadas_orientacion = []

        # Si hay resultados, extraemos las coordenadas de los puntos de referencia
        if resultados.multi_face_landmarks is not None:
            for puntos_de_referencia_cara in resultados.multi_face_landmarks:
                for indice in self.indices_ojo_izq:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ojo_izq.append((x, y))
                for indice in self.indices_ojo_der:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ojo_der.append((x, y))
                for indice in self.incide_central:
                    x = puntos_de_referencia_cara.landmark[indice].x
                    y = puntos_de_referencia_cara.landmark[indice].y
                    coordenadas_central = (x, y)
                for indice in self.indices_orientacion:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_orientacion.append((x, y))
                for indice in self.indices_ear_izq:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ear_izq.append((x, y))
                for indice in self.indices_ear_der:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ear_der.append((x, y))
            return coordenadas_ojo_izq, coordenadas_ojo_der, coordenadas_ear_izq, coordenadas_ear_der, coordenadas_central, coordenadas_orientacion
        return None


# ------------------------------------------ FUNCIONES DE CONTROL DEL EAR ------------------------------------------

    def calcular_ear(self, coordenadas_ojo):
        """
        Calcula el EAR de un ojo

        Args:
            coordenadas_ojo (list): Coordenadas de los puntos de referencia de un ojo
        
        Returns:
            float: EAR del ojo
        """
        d_A = np_linalg.norm(np_array(coordenadas_ojo[1]) - np_array(coordenadas_ojo[5]))
        d_B = np_linalg.norm(np_array(coordenadas_ojo[2]) - np_array(coordenadas_ojo[4]))
        d_C = np_linalg.norm(np_array(coordenadas_ojo[0]) - np_array(coordenadas_ojo[3]))
        ear = (d_A + d_B) / (2.0 * d_C)
        return ear


    def calcular_ear_medio(self, coordenadas_ojo_izquierdo, coordenadas_ojo_derecho):
        """
        Calcula el EAR medio de los dos ojos

        Args:
            coordenadas_ojo_izquierdo (list): Coordenadas de los puntos de referencia del ojo izquierdo
            coordenadas_ojo_derecho (list): Coordenadas de los puntos de referencia del ojo derecho
        
        Returns:
            float: EAR medio de los dos ojos
        """
        ear_izq = self.calcular_ear(coordenadas_ojo_izquierdo)
        ear_der = self.calcular_ear(coordenadas_ojo_derecho)
        ear_medio = (ear_izq + ear_der) / 2.0
        return ear_medio
            

#---------------------------   FUNCIONES DE CONTROL DE LAS DISTANCIAS    -------------------------------
    
    def calcular_distancias_ojos(self, coord_o_izq, coord_o_der):
        """
        Calcula las distancias entre la pupila y los puntos de referencia del ojo

        Args:
            coord_o_izq (list): Coordenadas de los puntos de referencia del ojo izquierdo
            coord_o_der (list): Coordenadas de los puntos de referencia del ojo derecho

        Returns:
            list: Distancias entre la pupila y los puntos de referencia de los ojos izquierdo y derecho
        """
        distancias_izq = []
        distancias_der = []
        for i in range(1, len(coord_o_izq)):
            distancias_izq.append(np_linalg.norm(np_array(coord_o_izq[0]) - np_array(coord_o_izq[i])))
        for i in range(1, len(coord_o_der)):
            distancias_der.append(np_linalg.norm(np_array(coord_o_der[0]) - np_array(coord_o_der[i])))
        return distancias_izq, distancias_der



    def calcular_eje_y(self, coord_ojo_izquierdo, coord_ojo_derecho):
        """
        Calcula la orientación de la cabeza en el eje y

        Args:
            coord_ojo_izquierdo (tuple): Coordenadas del ojo izquierdo
            coord_ojo_derecho (tuple): Coordenadas del ojo derecho

        Returns:
            float: Orientación de la cabeza en el eje y
        """
        # Calculamos la linea recta que une los dos ojos
        x1, y1 = coord_ojo_izquierdo
        x2, y2 = coord_ojo_derecho

        # Calculamos la pendiente de la recta
        m = (y2 - y1) / (x2 - x1)

        # Devolvemos la pendiente normalizada
        m = m*0.5 + 0.5

        return np_clip(m, 0, 1)



    def get_orientacion_cabeza(self, coord_o, size):
        """
        Funcion encargada de obtener la orientación de la cabeza en los ejes x, y, z

        Args:
            coord_o (list): Coordenadas de los puntos de referencia de la cara
            size (tuple): Tamaño de la imagen
        
        Returns:
            float: Orientación de la cabeza en el eje x
            float: Orientación de la cabeza en el eje y
            float: Orientación de la cabeza en el eje z
        """

        #Los puntos de referencia de la cabeza
        image_points = np_array([
            coord_o[0],     # Nariz
            coord_o[1],     # menton
            coord_o[2],     # ojo izquierdo
            coord_o[3],     # ojo derecho
            coord_o[4],     # esquina de la boca izquierda
            coord_o[5]      # esquina de la boca derecha
        ], dtype="double")  


        # Los puntos en 3D
        model_points = np_array([
            (0.0, 0.0, 0.0),       # nariz
            (0, -63.6, -12.5),     # Menton
            (-43.3, 32.7, -26),    # ojo izquierdo
            (43.3, 32.7, -26),     # ojo derecho
            (-28.9, -28.9, -24.1), # esquina de la boca izquierda
            (28.9, -28.9, -24.1)   # esquina de la boca derecha
        ])
        
        # Camara
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np_array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )
        dist_coeffs = np_zeros((4,1)) # Se da por hecho que no hay distorsion

        # Estimamos la pose de la cabeza
        (_, rotation_vector, translation_vector) = cv2_solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2_SOLVEPNP_ITERATIVE)

        # Convertimos el vector de rotación a ángulos de Euler 
        (rot_mat, _) = cv2_Rodrigues(rotation_vector)
        (_, _, _, _, _, _, euler_angles) = cv2_decomposeProjectionMatrix(cv2_hconcat((rot_mat, translation_vector)))
        
        # Normalizar a rango [0, 1]
        euler_angles_normalized = np_clip((euler_angles / 90) + 0.5, 0, 1)

        #Calculamos la horientacion en el eje y
        euler_angles_normalized[2,0] = self.calcular_eje_y(coord_o[2], coord_o[3])
        
        # Devolvemos los ángulos de Euler normalizados
        return euler_angles_normalized[0, 0], euler_angles_normalized[1, 0], euler_angles_normalized[2, 0]

