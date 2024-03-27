import mediapipe as mp
import numpy as np
import cv2


class Detector:
    def __init__(self):
        # Índices de los puntos de referencia para los ojos 
        # Primero las pupilas y despues el resto, asi el EAR siempre se calcula bien
        self.indices_ojo_izquierdo = [468, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  
        self.indices_ojo_derecho = [473, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        #Modelo de MediaPipe
        self.deteccion_cara = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,  
                                                            refine_landmarks=True,  #Para incluir la deteccion de las pupilas
                                                            min_detection_confidence=0.5)
        
        
    #Funcion principal de obtener coordeandas de los ojos
    def obtener_coordenadas(self, frame, ear):
        altura, ancho, _ = frame.shape              
        resultados = self.deteccion_cara.process(frame)  
        indices_ojo_izq = []
        indices_ojo_der = []
        coordenadas_ojo_izq = []
        coordenadas_ojo_der = [] 

        if ear:
            indices_ojo_izq = [33, 160, 158, 133, 153, 144]
            indices_ojo_der = [362, 385, 387, 263, 373, 380]
        else:
            indices_ojo_izq = self.indices_ojo_izquierdo
            indices_ojo_der = self.indices_ojo_derecho

        if resultados.multi_face_landmarks is not None:
            for puntos_de_referencia_cara in resultados.multi_face_landmarks:
                for indice in indices_ojo_izq:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ojo_izq.append((x, y))
                for indice in indices_ojo_der:
                    x = int(puntos_de_referencia_cara.landmark[indice].x * ancho)
                    y = int(puntos_de_referencia_cara.landmark[indice].y * altura)
                    coordenadas_ojo_der.append((x, y))
        return coordenadas_ojo_izq, coordenadas_ojo_der




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
    def calcular_distancias_ojo(self, coordenadas_ojo):
        distancias = []
        for i in range(1, len(coordenadas_ojo)):
            distancias.append(np.linalg.norm(np.array(coordenadas_ojo[0]) - np.array(coordenadas_ojo[i])))
        return distancias
    

    #Funcion para calcular las distancias de los dos ojos con las pupilas
    def calcular_distancias_ojos(self, coordenadas_ojo_izquierdo, coordenadas_ojo_derecho):
        distancias_izq = self.calcular_distancias_ojo(coordenadas_ojo_izquierdo)
        distancias_der = self.calcular_distancias_ojo(coordenadas_ojo_derecho)
        return distancias_izq, distancias_der  
    

    # Funcion para medir el largo de los ojos medio
    def calcular_medida_ojo_media(self, distancias_izq, distancias_der):
        medida_ojo_izq = np.mean(distancias_izq)
        medida_ojo_der = np.mean(distancias_der)
        medida_ojo_media = (medida_ojo_izq + medida_ojo_der) / 2.0
        return medida_ojo_media
    
    # Funcion para detectar la posicion de la cabeza en la pantalla
    #(La coordenada x,y del punto 8 normalizada entre 0 y 1 con el tamaño del frame)
    def obtener_posicion_cabeza(self, frame):
        altura, ancho, _ = frame.shape
        resultados = self.deteccion_cara.process(frame)
        if resultados.multi_face_landmarks is not None:
            for puntos_de_referencia_cara in resultados.multi_face_landmarks:
                x = puntos_de_referencia_cara.landmark[8].x
                y = puntos_de_referencia_cara.landmark[8].y
                return x, y
        return None
    

    def get_orientacion_cabeza(self, frame):
        # Obtenemos los resultados del modelo de detección de cara
        resultados = self.deteccion_cara.process(frame)

        # Si se detectó una cara
        if resultados.multi_face_landmarks is not None:
            # Obtenemos los puntos de referencia de la cara
            puntos_de_referencia_cara = resultados.multi_face_landmarks[0]

            def _relative(punto, shape):
                altura, ancho = shape[0], shape[1]
                x = int(punto.x * ancho)
                y = int(punto.y * altura)
                return (x, y)

            # Obtenemos los puntos de imagen 2D
            image_points = np.array([
                _relative(puntos_de_referencia_cara.landmark[4], frame.shape),    # Nose tip
                _relative(puntos_de_referencia_cara.landmark[152], frame.shape),  # Chin
                _relative(puntos_de_referencia_cara.landmark[263], frame.shape),  # Left eye left corner
                _relative(puntos_de_referencia_cara.landmark[33], frame.shape),   # Right eye right corner
                _relative(puntos_de_referencia_cara.landmark[287], frame.shape),  # Left Mouth corner
                _relative(puntos_de_referencia_cara.landmark[57], frame.shape)    # Right mouth corner
            ], dtype="double")

            # 3D model points.
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            # Camera internals
            size = frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double"
            )

            dist_coeffs = np.zeros((4,1)) # Asumiendo que no hay distorsion

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

        # Si no se detectó una cara, devolvemos None
        return None
