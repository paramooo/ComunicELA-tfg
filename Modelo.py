from Detector import Detector
from Camara import Camara
from kivy.uix.label import Label
import pygame
import random
import numpy as np
import os
from kivy.app import App
from Mensajes import Mensajes


class Modelo:
    def __init__(self):
        # Para el sonido del click
        pygame.mixer.init()

        # Se inicializa el detector
        self.detector = Detector()
        self.camara = Camara()
        self.iniciar_camara()

        # Variables de control para el tamaño de la fuente de los textos
        self.tamaño_fuente_txts = 19

        # Variables de control para la calibracion del parpadeo
        self.estado_calibracion = 0
        self.umbral_ear = 0.2
        self.umbral_ear_bajo = 0.2
        self.umbral_ear_cerrado = 0.2
        self.contador_p = 0
        self.suma_frames = 7
        self.calibrado = False
        self.sonido = pygame.mixer.Sound('sonidos/click.wav')


        # Variables para la recopilacion de datos 
        self.reiniciar_datos_r()
        self.input = []
        self.output = []

        
    #Funcion para reiniciar los datos despues de cada escaneo (se aprovecha para inicializarlos tambien)
    def reiniciar_datos_r(self):
        self.recopilar = False
        self.contador_r = 5
        self.pos_r = (0, 0)
        self.salto_bajo, self.salto_alto = 30, 80
        self.velocidad = 50
        self.direccion = 1

    
# ---------------------------   FUNCIONES DE CONTROL GENERAL    -------------------------------
#-------------------------------------------------------------------------------------------
        
    def mensaje(self, mensaje):
        # Crear el mensaje de mensaje
        mensaje = Mensajes(mensaje)

        # Agregar el mensaje a la pantalla actual
        App.get_running_app().root.current_screen.add_widget(mensaje)




# ---------------------------   FUNCIONES DE CONTROL DE LA CAMARA    -------------------------------
#-------------------------------------------------------------------------------------------

    def iniciar_camara(self):
        # Se inicia el escaneo de los ojos
        self.camara.start()

    def detener_camara(self):
        # Se detiene el escaneo de los ojos
        self.camara.stop()

    def camara_activa(self):
        return self.camara.camara_activa()
    
    def get_frame(self):
        return self.camara.frame




# ---------------------------   FUNCIONES CONTROL DEL MENU DE CALIBRACION -------------------------------
#-------------------------------------------------------------------------------------------

    def cambiar_estado_calibracion(self, numero):
        if numero != -1:
            self.estado_calibracion = numero
        else:
            # Se cambia al siguiente estado
            self.estado_calibracion = self.estado_calibracion + 1
        return self.estado_calibracion
                
    def obtener_estado_calibracion(self):
        return self.estado_calibracion        


# ---------------------------   FUNCIONES DE CONTROL DEL EAR -------------------------------
#-------------------------------------------------------------------------------------------
    
    def capturar_ear(self, frame):
        #Si aun no hay frame, se devuelve 0
        if frame is None:
            return 1
        
        # Se obtienen las coordenadas de los ojos de los indices de los ear
        coordeanadas_ojo_izquierdo_ear, coordeanadas_ojo_derecho_ear = self.detector.obtener_coordenadas(frame, ear=True)
        
        # Si no se detecta cara, se devuelve 0
        if len(coordeanadas_ojo_izquierdo_ear) == 0 or len(coordeanadas_ojo_derecho_ear) == 0:
            return 1
        
        # Si sale bien se calcula el EAR medio
        ear_medio = self.detector.calcular_ear_medio(coordeanadas_ojo_izquierdo_ear, coordeanadas_ojo_derecho_ear)
        return ear_medio


    def calibrar_ear(self, frame):
        # Se captura el EAR actual
        if self.estado_calibracion == 0:
            self.umbral_ear_bajo = self.capturar_ear(frame)
        elif self.estado_calibracion == 1:
            self.umbral_ear_cerrado = self.capturar_ear(frame)
            self.umbral_ear = (self.umbral_ear_bajo*0.3 + self.umbral_ear_cerrado*0.7) #Se calcula el umbral final ponderado entre el cerrado y el abierto bajo
            self.calibrado = True

            
    def get_parpadeo(self):
        frame = self.camara.frame
        ear = self.capturar_ear(frame)
        if ear < self.umbral_ear:
            self.contador_p += 1
            if self.contador_p == self.suma_frames:
                self.contador_p = -50
                self.sonido.play()
            return 1
        else:
            self.contador_p = 0     
            return 0   


# ---------------------------   FUNCIONES DE DISTANCIAS -------------------------------
            
    #Coge las distancias de los ojos con respecto a la pupila de los indices de Detector
    def get_distancias_ojos(self, frame):
        if frame is None:
            return 
        coordenadas_izq, coordenadas_der =  self.detector.obtener_coordenadas(frame, ear=False)
        if len(coordenadas_izq) == 0 or len(coordenadas_der) == 0:
            return
        distancias_izq, distancias_der = self.detector.calcular_distancias_ojos(coordenadas_izq, coordenadas_der)
        return distancias_izq, distancias_der
    

    def get_medida_ojo_media(self, distancias):
        return self.detector.calcular_medida_ojo_media(distancias[0], distancias[1])



    

# ---------------------------   FUNCIONES DE RECOPILACION DE DATOS  -------------------------------
        
    def cuenta_atras(self, dt):
        if self.contador_r > 0:
            self.contador_r -= 1
        elif self.contador_r == 0:
            self.recopilar = True
            return False

    def actualizar_pos_circle_r(self, tamano_pantalla):
        # Actualiza la posición x de la pelota
        self.pos_r = (self.pos_r[0] + self.velocidad * self.direccion, self.pos_r[1])

        # Si la pelota toca los bordes x de la pantalla, cambia la dirección y realiza un salto
        if self.pos_r[0] < 0 or self.pos_r[0] + 50 > tamano_pantalla[0]:
            self.direccion *= -1

            # Actualiza la posición y de la pelota con un salto aleatorio
            salto = random.randint(self.salto_bajo, self.salto_alto)
            self.pos_r = (self.pos_r[0], self.pos_r[1] + salto)

        # Si la pelota toca el borde superior de la pantalla, invierte el salto
        if self.pos_r[1] + 50 > tamano_pantalla[1]:
            self.salto_bajo, self.salto_alto = -self.salto_alto, -self.salto_bajo
            self.pos_r = (self.pos_r[0], tamano_pantalla[1] - 50)

        # Si la pelota toca el borde inferior de la pantalla, reiniciamos los datos y la posición
        if self.pos_r[1] < 0:
            self.guardar_final()
            self.reiniciar_datos_r()
        else:
            # Aquí se mandarán los datos a otra función para que los guarde en el txt junto con las distancias, etc.
            frame = self.camara.frame

            if frame is not None:
                # DATOS A RECOPILAR:
                # - Distancias de los ojos - ok
                distancias = self.get_distancias_ojos(frame)

                # - Medida del ojo medio - ok
                medida_ojo_media = None
                # Al comprobar asi el None, nos aseguramos de que todos los datos son validos
                if distancias is None or distancias[0] is None or distancias[1] is None:
                    distancias = None
                else:
                    medida_ojo_media = self.detector.calcular_medida_ojo_media(distancias[0], distancias[1])


                # - Orientación de la cabeza - ok
                orientacion_cabeza = self.detector.get_orientacion_cabeza(frame)


                # - EAR - ok
                #Este si no detecta cara marca 1, no hace falta asegurarse de que sea valido
                ear = self.capturar_ear(frame)


                # - Posición de la cabeza en el frame - ok
                pos_cabeza = self.detector.obtener_posicion_cabeza(frame)

                # Si todos los datos son validos, se guardan junto con las coordenadas de la pelota
                if any(element is None for element in [distancias, medida_ojo_media, orientacion_cabeza, ear, pos_cabeza, self.pos_r]):
                    pass
                else:
                    self.guardar_datos(distancias, medida_ojo_media, orientacion_cabeza, ear, pos_cabeza, self.pos_r/np.array(tamano_pantalla))
        return self.pos_r



    def guardar_datos(self, distancias, medida_ojo_media, orientacion_cabeza, ear, pos_cabeza, pos_r):
        # Preparar los datos para guardar
        distancias_str = ', '.join([str(dist) for dist in distancias[0] + distancias[1]])
        medida_ojo_media_str = str(medida_ojo_media)
        orientacion_cabeza_str = ', '.join([str(coord) for coord in orientacion_cabeza])
        ear_str = str(ear)
        ear_umbral_str = str(self.umbral_ear) 
        pos_cabeza_str = ', '.join([str(coord) for coord in pos_cabeza])

        # Guardar los datos en las listas
        self.input.append(f'{distancias_str}, {medida_ojo_media_str}, {orientacion_cabeza_str}, {pos_cabeza_str}, {ear_str}, {ear_umbral_str}')
        self.output.append(f'{pos_r[0]}, {pos_r[1]}')





    def guardar_final(self):
        os.makedirs('txts', exist_ok=True)

        # Guardar los datos en los archivos
        with open('txts/input.txt', 'a') as f:
            for linea in self.input:
                f.write(linea + '\n')

        with open('txts/output.txt', 'a') as f:
            for linea in self.output:
                f.write(linea + '\n')

        # Limpiar las listas para la próxima vez
        self.input = []
        self.output = []