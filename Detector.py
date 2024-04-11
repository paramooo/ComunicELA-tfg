import mediapipe as mp
import numpy as np
import cv2

class Detector:
    def __init__(self):
        # Índices de los puntos de referencia para los ojos 
        # Primero las pupilas y despues el resto, asi el EAR siempre se calcula bien
        self.indices_ojo_izq = [473, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]  
        self.indices_ojo_der = [468, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.indices_ear_izq = [33, 160, 158, 133, 153, 144]
        self.indices_ear_der = [362, 385, 387, 263, 373, 380]
        self.incide_central = [8]
        self.indices_orientacion = [4, 152, 263, 33, 287, 57]

        #Modelo de MediaPipe
        self.deteccion_cara = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,  
                                                            refine_landmarks=True,  #Para incluir la deteccion de las pupilas
                                                            min_detection_confidence=0.5)
        
        
    #Funcion principal de obtener coordeandas de los ojos
    def obtener_coordenadas_indices(self, frame):
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




#---------------------------   FUNCIONES DE CONTROL DEL EAR    -------------------------------
    
    #Funcion individual para cada ojo
    def calcular_ear(self, coordenadas_ojo):
        d_A = np.linalg.norm(np.array(coordenadas_ojo[1]) - np.array(coordenadas_ojo[5]))
        d_B = np.linalg.norm(np.array(coordenadas_ojo[2]) - np.array(coordenadas_ojo[4]))
        d_C = np.linalg.norm(np.array(coordenadas_ojo[0]) - np.array(coordenadas_ojo[3]))
        ear = (d_A + d_B) / (2.0 * d_C)
        return ear

    # Funcion para calcular el ear medio de los dos ojos
    def calcular_ear_medio(self, coordenadas_ojo_izquierdo, coordenadas_ojo_derecho):
        ear_izq = self.calcular_ear(coordenadas_ojo_izquierdo)
        ear_der = self.calcular_ear(coordenadas_ojo_derecho)
        ear_medio = (ear_izq + ear_der) / 2.0
        return ear_medio
            

#---------------------------   FUNCIONES DE CONTROL DE LAS DISTANCIAS    -------------------------------
    
    #Funcion para calcular las distancias entre los puntos y la pupila
    #La pupila es el primer punto de la lista de coordenadas
    def calcular_distancias_ojos(self, coord_o_izq, coord_o_der):
        distancias_izq = []
        distancias_der = []
        for i in range(1, len(coord_o_izq)):
            distancias_izq.append(np.linalg.norm(np.array(coord_o_izq[0]) - np.array(coord_o_izq[i])))
        for i in range(1, len(coord_o_der)):
            distancias_der.append(np.linalg.norm(np.array(coord_o_der[0]) - np.array(coord_o_der[i])))
        return distancias_izq, distancias_der

    

    # Funcion para detectar la posicion de la cabeza en la pantalla
    def get_orientacion_cabeza(self, coord_o, frame):
            #Los puntos de referencia de la cabeza
            image_points = np.array([
                coord_o[0],     # Nose tip
                coord_o[1],     # Chin
                coord_o[2],     # Left eye left corner
                coord_o[3],     # Right eye right corner
                coord_o[4],     # Left Mouth corner
                coord_o[5]      # Right mouth corner
            ], dtype="double")  


            # Los puntos en 3D
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            # Camara
            size = frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
            )
            dist_coeffs = np.zeros((4,1)) # Se da por hecho que no hay distortion

            # Estimamos la pose de la cabeza
            (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # Convertimos el vector de rotación a ángulos de Euler para obtener la orientación de la cabeza
            (rot_mat, _) = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rot_mat, translation_vector))
            (_, _, _, _, _, _, euler_angles) = cv2.decomposeProjectionMatrix(pose_mat)

            # Normalizamos los ángulos de Euler al rango [0, 1]
            euler_angles_normalized = (euler_angles / 90) + 0.5

            # Nos aseguramos de que los ángulos normalizados estén en el rango [0, 1]
            euler_angles_normalized = np.clip(euler_angles_normalized, 0, 1)

            return euler_angles_normalized[0, 0], euler_angles_normalized[1, 0]  # Devolvemos los ángulos de Euler normalizados para la orientación de la cabeza
    


    # ------------------- FUNCIONES PROPIAS DE CONJUNTOS --------------------------
    #CONJUNTO 1----------
    # Funcion para medir el largo de los ojos medio
    def calcular_medida_ojo_media(self, distancias_izq, distancias_der):
        medida_ojo_izq = np.mean(distancias_izq)
        medida_ojo_der = np.mean(distancias_der)
        medida_ojo_media = (medida_ojo_izq + medida_ojo_der) / 2.0
        return medida_ojo_media
    

    #CONJUNTO 3----------