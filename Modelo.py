from Detector import Detector
from Camara import Camara
import pygame
import random
import numpy as np
import os
from kivy.app import App
from Mensajes import Mensajes
from entrenamiento.Conjuntos import Conjuntos
import cv2
import torch
from gtts import gTTS
from io import BytesIO
import pandas as pd
from kivy.core.audio import SoundLoader
from pykalman import KalmanFilter
class Modelo:
    def __init__(self):
        # Para el sonido del click
        pygame.mixer.init()

        # Se inicializa el detector
        self.detector = Detector()
        self.camara = Camara()
        self.camara_act = None

        # Variables para el modelo de test
        self.conjunto = 2
        self.modelo = torch.load('./anns/pytorch/modelo.pth')

        # Variables de control para el tamaño de la fuente de los textos
        self.tamaño_fuente_txts = 23

        # Variables de control para la calibracion del parpadeo
        self.estado_calibracion = 0
        self.umbral_ear = 0.2
        self.umbral_ear_bajo = 0.2
        self.umbral_ear_cerrado = 0.2
        self.contador_p = 0
        self.suma_frames = 4 #Numero de frames que tiene que estar cerrado el ojo para que se considere un parpadeo
        self.calibrado = False
        self.sonido = pygame.mixer.Sound('./sonidos/click.wav')


        # Variables para la recopilacion de datos 
        self.reiniciar_datos_r()
        self.input = []
        self.output = []


        # Variable para el modelo
        self.pos_t = (0, 0)
        self.escanear = False

        # Variables para suavizar la mirada en el test
        self.historial = []
        #self.cantidad_suavizado = 50
        self.hist_max = 15
        #self.retroceso_click = 0

        # Variables para uso de los tableros
        self.tableros = {}
        self.cargar_tableros()
        self.frase = ""
        self.tablero = None
        
    #Funcion para reiniciar los datos despues de cada escaneo (se aprovecha para inicializarlos tambien)
    def reiniciar_datos_r(self):
        self.recopilar = False #Variable para saber si se esta recopilando datos
        self.contador_r = 5 #Contador para la cuenta atras
        self.pos_r = (0, 0) #Posicion de la pelota roja
        self.salto_bajo, self.salto_alto = 30, 80 #Salto de la pelota roja
        self.velocidad = 30
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
        self.camara_act = camara

    def get_index_actual(self):
        return self.camara_act

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
    
    def get_frame_editado(self, porcentaje):
        frame = self.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Poner un texto de que no esta la camara activa alineado al centro y en blanco y con la fuente de texto
            cv2.putText(frame, 'Seleccione una camara', (220, 430), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
            return frame
            
        
        # Rojo por defecto
        color = (0, 0, 255)  
        
        # Poner el punto en el centro
        coord_central = self.detector.get_punto_central(frame)
        puntos_or = self.detector.get_puntos_or(frame)
        if coord_central is not None:
            x = round(frame.shape[1]*coord_central[0])
            y = round(frame.shape[0]*coord_central[1])

            # Si el punto central esta en el centro de la pantalla del tamaño del porcentaje esta en verde el punto central y en rojo si no
            if 0.5 - porcentaje/2 < coord_central[0] < 0.5 + porcentaje/2 and 0.5 - porcentaje/2 < coord_central[1] < 0.5 + porcentaje/2:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            frame[y - 4:y + 4, x - 4:x + 4, :] = color

        if puntos_or is not None:
            for punto in puntos_or:
                x = round(frame.shape[1]*punto[0])
                y = round(frame.shape[0]*punto[1])
                frame[y - 1:y + 1, x - 1:x + 1, :] = color

        # Poner los pixeles centrales del frame para linea horizontal de la cruz
        frame[frame.shape[0]//2 - 1:frame.shape[0]//2 + 1, :, :] = color
        # Poner los pixeles centrales del frame para linea vertical de la cruz
        frame[:, frame.shape[1]//2 - 1:frame.shape[1]//2 + 1, :] = color
        
        # Calcular las coordenadas del cuadrado
        x_start = int(frame.shape[1] * (0.5 - porcentaje / 2))
        x_end = int(frame.shape[1] * (0.5 + porcentaje / 2))
        y_start = int(frame.shape[0] * (0.5 - porcentaje / 2))
        y_end = int(frame.shape[0] * (0.5 + porcentaje / 2))

        # Dibujar el cuadrado en el frame
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

        return frame





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
            self.umbral_ear = (self.umbral_ear_bajo*0.4 + self.umbral_ear_cerrado*0.6) #Se calcula el umbral final ponderado entre el cerrado y el abierto bajo
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
            return 0
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

    def actualizar_pos_circle_r(self, tamano_pantalla, fichero):
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
            self.guardar_final(fichero)
            self.reiniciar_datos_r()
        else:
            datos = self.obtener_datos()
            if datos is None:                
                self.mensaje("No se detecta cara")	
            else:
                distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab = datos
                self.guardar_datos(distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, self.pos_r/np.array(tamano_pantalla))
        return self.pos_r



    def guardar_datos(self, distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, pos_r_norm):
        # Preparar los datos para guardar
        distancias_izq_str = ', '.join([str(dist) for dist in distancias_izq])
        distancias_der_str = ', '.join([str(dist) for dist in distancias_der])
        orientacion_cabeza_str = f'{or_x}, {or_y}, {or_z}'
        ear_str = str(ear)
        ear_umbral_str = str(umbral_ear) 
        pos_cabeza_str = ', '.join([str(coord) for coord in coord_cab])

        # Guardar los datos en las listas
        self.input.append(f'{distancias_izq_str}, {distancias_der_str}, {orientacion_cabeza_str}, {pos_cabeza_str}, {ear_str}, {ear_umbral_str}')
        self.output.append(f'{pos_r_norm[0]}, {pos_r_norm[1]}')



    def guardar_final(self, fichero):
        #Si no existe la carpeta txts, se crea
        os.makedirs('txts', exist_ok=True)
        
        # Guardar los datos en los archivos
        with open(f'./txts/input{fichero}.txt', 'a') as f:
            for linea in self.input:
                f.write(linea + '\n')

        with open(f'./txts/output{fichero}.txt', 'a') as f:
            for linea in self.output:
                f.write(linea + '\n')

        # Limpiar las listas para la próxima vez
        self.input = []
        self.output = []


# ---------------------------   FUNCIONES DE OBTENCION DE DATOS  -------------------------------
#-----------------------------------------------------------------------------------------------
    def obtener_datos(self):
        frame = self.get_frame()
        if frame is None:
            self.mensaje("Error de la camara")
            return None
        datos = self.detector.obtener_coordenadas_indices(frame)
        
        #Si no se detecta cara, se devuelve None
        if datos is None:
            self.mensaje("No se detecta cara")
            return None
        
        # Se desempaquetan los datos
        coord_o_izq, coord_o_der, coord_ear_izq, coord_ear_der, coord_cab, coord_o = datos

        # - Distancias de los ojos 
        distancias_izq, distancias_der = self.detector.calcular_distancias_ojos(coord_o_izq, coord_o_der)

        # - Orientación de la cabeza entre 0 y 1
        or_x, or_y, or_z = self.detector.get_orientacion_cabeza(coord_o, frame)

        # - EAR 
        ear = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        #print("ORX: ", round(or_x,3), "ORY: ", round(or_y,3), "ORZ: ", round(or_z,3), "coord_cab: ", coord_cab)

        # Pasamos la posicion de la pantalla normalizada
        return distancias_izq, distancias_der, or_x, or_y, or_z, ear, self.umbral_ear, coord_cab



    #Funcion para obtener la posicion de la mirada en el test
    #Se obtiene la posición de la mirada y si se ha hecho click
    #Se suaviza la posición de la mirada y si se hace click, se retrocede self.retroceso_click frames para evitar la desviación
    #el error esta aqui en como se le pasan los datos en el datos as array creo 
    def obtener_posicion_mirada_ear(self):
        # Se obtienen los datos
        datos = self.obtener_datos()

        # Si no se detecta cara, se devuelve None, None
        if datos is None:
            return None

        # Se desempaquetan los datos del ear para el click
        _, _, _, _, _, ear, _, _ = datos
        click = self.get_parpadeo(ear)

        # Se transforman los datos a un conjunto
        datos_array = Conjuntos.datos_as_array(datos)

        # Normalizar los datos
        normalizar_funcion = getattr(Conjuntos, f'conjunto_{self.conjunto}')
        datos_array = normalizar_funcion(datos_array)


        # Normalizacion para la ResNet
        #----------------------------------
        # datos_array = np.reshape(datos_array, (-1, 1, 23))
        # datos_array = np.repeat(datos_array[:, :, np.newaxis], 3, axis=1)
        # self.modelo.eval()
        #----------------------------------


        datos_array = torch.from_numpy(datos_array).float()

        # Se predice la posición de la mirada
        mirada = self.modelo(datos_array)

        # Se desempaqueta la posición de la mirada
        mirada = mirada.data.numpy()[0]

        # Añadir la nueva posición al historial
        self.historial.append(mirada)

        # Eliminar la posición más asntigua si el historial es demasiado largo
        if len(self.historial) > self.hist_max:
            self.historial.pop(0)

        #Mediana -> funciona bastante bien (menos retraso quel normal pero aun tiene)
        mirada = np.median(self.historial[-self.hist_max:], axis=0)


        # Calcular la media ponderada de las posiciones en el historial para suavizar
        # if len(self.historial) > self.cantidad_suavizado:
        #     pesos = range(1, len(self.historial[-self.cantidad_suavizado:]) + 1)
        #     mirada = np.average(self.historial[-self.cantidad_suavizado:], weights=pesos, axis=0)

        # Hacer la media normal con el historial
        mirada = np.mean(self.historial, axis=0)

        # ------------------ Al hacerlo con lo del bloqueo ya no hace falta esto -------------------
        # Si se detecta un parpadeo, se coge la posición de self.retroceso_click frames atrás
        # if click == 1 and len(self.historial) >= self.retroceso_click:
        #     mirada = self.historial[-self.retroceso_click]

        return mirada, click


                
   
#--------------------------------FUNCIONES PARA LOS TABLEROS--------------------------------
#-------------------------------------------------------------------------------------------

    # def cargar_tableros(self):
    #     for filename in os.listdir('./tableros'):
    #         if filename.endswith('.txt'):
    #             with open(os.path.join('./tableros', filename), 'r') as f:
    #                 palabras = [line.strip().split(';') for line in f]
    #                 self.tableros[filename[:-4]] = palabras  # Añade el tablero al diccionario


    def cargar_tableros(self):
        for filename in os.listdir('./tableros'):
            if filename.endswith('.xlsx'):
                df = pd.read_excel(os.path.join('./tableros', filename), header=None)
                palabras = df.values.tolist()  # Convierte el DataFrame a una lista de listas
                self.tableros[filename[:-5]] = palabras  # Añade el tablero al diccionario

    def obtener_tablero(self, nombre):
        return self.tableros.get(nombre.lower())
    
    def get_frase(self):
        return self.frase
    
    def añadir_palabra(self, palabra):
        self.frase += palabra + ' '

    def borrar_palabra(self):
        self.frase = ' '.join(self.frase.rstrip().split(' ')[:-1]) + ' '

    def borrar_todo(self):
        self.frase = ''

    def reproducir_texto(self):
        try:
            tts = gTTS(text=self.frase.lower(), lang='es')    
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()

        except:
            print("Error al reproducir el texto")
            pass
