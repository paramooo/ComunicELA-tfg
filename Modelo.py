from Detector import Detector
from Camara import Camara
from kivy.uix.label import Label
import pygame
import random
import numpy as np
import os
from kivy.app import App
from Mensajes import Mensajes
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class Modelo:
    def __init__(self):
        # Para el sonido del click
        pygame.mixer.init()

        # Se inicializa el detector
        self.detector = Detector()
        self.camara = Camara()

        # Variables de control para el tamaño de la fuente de los textos
        self.tamaño_fuente_txts = 23

        # Variables de control para la calibracion del parpadeo
        self.estado_calibracion = 0
        self.umbral_ear = 0.2
        self.umbral_ear_bajo = 0.2
        self.umbral_ear_cerrado = 0.2
        self.contador_p = 0
        self.suma_frames = 1
        self.calibrado = False
        self.sonido = pygame.mixer.Sound('./sonidos/click.wav')


        # Variables para la recopilacion de datos 
        self.reiniciar_datos_r()
        self.input = []
        self.output = []


        # Variable para el modelo 
        self.modelo = tf.keras.models.load_model('./anns/ann_conj3_20k.keras')
        self.pos_t = (0, 0)
        self.escanear = False

        # Variables para suavizar el movimiento del circulo de la salida de la red neuronal
        self.historial = []
        self.cantidad_suavizado = 5

        
    #Funcion para reiniciar los datos despues de cada escaneo (se aprovecha para inicializarlos tambien)
    def reiniciar_datos_r(self):
        self.recopilar = False
        self.contador_r = 5
        self.pos_r = (0, 0)
        self.salto_bajo, self.salto_alto = 30, 80
        self.velocidad = 25
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

    def iniciar_camara(self, index):
        # Se inicia el escaneo de los ojos
        self.camara.start(index)

    def detener_camara(self):
        # Se detiene el escaneo de los ojos
        self.camara.stop()

    def camara_activa(self):
        return self.camara.camara_activa()
    
    def get_frame(self):
        return self.camara.get_frame()
    
    def obtener_camaras(self):
        return self.camara.obtener_camaras()
    
    def seleccionar_camara(self, camara):
        if self.camara_activa():
            self.detener_camara()
        self.iniciar_camara(camara)



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
    
#Para la calibracion
    def calibrar_ear(self):
        #Si estamos en el estado 2, no se hace nada, sale al inicio
        if self.estado_calibracion == 2:
            return

        # Se obtienen los datos
        datos = self.detector.obtener_coordenadas_indices(self.get_frame())

        #Si no se detecta cara, se devuelve 1 indicando error
        if datos is None:
            return 1
    
        # Se desempaquetan los datos
        _, _, coord_ear_izq, coord_ear_der, _, _ = self.detector.obtener_coordenadas_indices(self.get_frame())

        # Se captura el EAR actual dependiendo del estado de la calibracion
        if self.estado_calibracion == 0:
            self.umbral_ear_bajo = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        elif self.estado_calibracion == 1:
            self.umbral_ear_cerrado = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)
            self.umbral_ear = (self.umbral_ear_bajo*0.5 + self.umbral_ear_cerrado*0.5) #Se calcula el umbral final ponderado entre el cerrado y el abierto bajo
            self.calibrado = True


#Para el test/tableros
#Reproducimos sonido pasado un tiempo de parpadeo pero el color en tiempo real
    def get_parpadeo(self, ear):
        if ear < self.umbral_ear:
            self.contador_p += 1
            if self.contador_p == self.suma_frames:
                self.contador_p = -1000 #Asi evitamos dos toques consecutivos sin abrir el ojo
                self.sonido.play()
            return 1
        else:
            self.contador_p = 0     
            return 0   


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
            datos = self.obtener_datos("recop")
            if datos is None:                
                self.mensaje("No se detecta cara")	
            else:
                distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab, self.pos_r = datos
                self.guardar_datos(distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab, self.pos_r/np.array(tamano_pantalla))
        return self.pos_r



    def guardar_datos(self, distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab, pos_r):
        # Preparar los datos para guardar
        distancias_izq_str = ', '.join([str(dist) for dist in distancias_izq])
        distancias_der_str = ', '.join([str(dist) for dist in distancias_der])
        medida_ojo_media_str = str(medida_ojo_media)
        orientacion_cabeza_str = f'{or_x}, {or_y}'
        ear_str = str(ear)
        ear_umbral_str = str(self.umbral_ear) 
        pos_cabeza_str = ', '.join([str(coord) for coord in coord_cab])

        # Guardar los datos en las listas
        self.input.append(f'{distancias_izq_str}, {distancias_der_str}, {medida_ojo_media_str}, {orientacion_cabeza_str}, {pos_cabeza_str}, {ear_str}, {ear_umbral_str}')
        self.output.append(f'{pos_r[0]}, {pos_r[1]}')



    def guardar_final(self):
        #Si no existe la carpeta txts, se crea
        os.makedirs('txts', exist_ok=True)

        # Guardar los datos en los archivos
        with open('./txts/input.txt', 'a') as f:
            for linea in self.input:
                f.write(linea + '\n')

        with open('./txts/output.txt', 'a') as f:
            for linea in self.output:
                f.write(linea + '\n')

        # Limpiar las listas para la próxima vez
        self.input = []
        self.output = []


# ---------------------------   FUNCIONES DE OBTENCION DE DATOS  -------------------------------
    def obtener_datos(self, modo):
        frame = self.get_frame()
        datos = self.detector.obtener_coordenadas_indices(frame)
        
        #Si no se detecta cara, se devuelve None
        if datos is None:
            return None
        
        # Se desempaquetan los datos
        coord_o_izq, coord_o_der, coord_ear_izq, coord_ear_der, coord_cab, coord_o = datos

        # - Distancias de los ojos 
        distancias_izq, distancias_der = self.detector.calcular_distancias_ojos(coord_o_izq, coord_o_der)

        # - Medida del ojo medio 
        medida_ojo_media = self.detector.calcular_medida_ojo_media(distancias_izq, distancias_der)


        # - Orientación de la cabeza entre 0 y 1
        or_x, or_y = self.detector.get_orientacion_cabeza(coord_o, frame)

        # - EAR 
        ear = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        # Pasamos la posicion de la pantalla normalizada
        if modo == "recop":
            return distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab, self.pos_r
        elif modo == "test":
            return distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab/np.array(frame.shape[:2])



# ---------------------------   FUNCIONES DE CONTROL DEL MOVIMIENTO DEL CIRCULO EN TEST  -------------------------------
#-----------------------------------------------------------------------------------------------------------------------
        
    def obtener_posicion_mirada_ear(self):
        click = 0

        # Se obtienen los datos
        datos = self.obtener_datos("test")

        #Si no se detecta cara, se devuelve None, None, None, None, None, None
        if datos is None:
            return None
        
        # Se desempaquetan los datos del ear para el click
        _, _, _, _, _, ear, _ = datos
        click = self.get_parpadeo(ear)
        
        # Se transforman los datos de entrada para el modelo
        entrada = self.transformar_a_conjunto3(datos)

        # Se predice la posición de la mirada
        mirada = self.modelo.predict(entrada)

        # Añadir la nueva posición al historial
        self.historial.append(mirada)

        # Eliminar la posición más antigua si el historial es demasiado largo
        if len(self.historial) > self.cantidad_suavizado:
            self.historial.pop(0)

        # Calcular la media ponderada de las posiciones en el historial
        pesos = range(1, len(self.historial) + 1) 
        mirada_suavizada = np.average(self.historial, weights=pesos, axis=0)

        #Aqui cambiar entre mirada suavizada y mirada para ver los resultados reales de la red
        return mirada_suavizada, click
 
            
    def transformar_a_conjunto3(self, datos):
        distancias_izq, distancias_der, medida_ojo_media, or_x, or_y, ear, coord_cab = datos

        # Calcular la media de las distancias de los dos ojos
        dist = [(i + j) / 2 for i, j in zip(distancias_izq, distancias_der)]

        # Normaliza las distancias con la medida del ojo medio
        dist = [distancia / medida_ojo_media for distancia in dist]

        # Concatenar todos los datos en un solo array y ordenadas como el conjunto 3
        datos_transformados = np.concatenate([dist, [or_x], [or_y], coord_cab, [ear], [self.umbral_ear]])

        # Añadir una dimensión para que sea compatible con el modelo
        datos_transformados = np.expand_dims(datos_transformados, axis=0)
        
        # Redondear los datos a 10 decimales 
        datos_transformados = np.round(datos_transformados, 3)

        return datos_transformados
