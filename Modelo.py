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
import threading
import math
from torch import nn
import torch.optim as optim
from kivy.clock import Clock
import json
import copy

class Modelo:
    def __init__(self):
        # Para el sonido del click
        pygame.mixer.init()

        # Se inicializa el detector
        self.detector = Detector()
        self.camara = Camara()
        self.camara_act = None
        self.desarrollador = False

        # Variables para el modelo de test
        self.conjunto = 2
        self.modelo_org = torch.load('./anns/pytorch/modelo_ajustado.pth')
        self.modelo = copy.deepcopy(self.modelo_org)

        # Variables de control para el tamaño de la fuente de los textos
        self.tamaño_fuente_txts = 25

        # Variables de control para la calibracion del parpadeo
        self.estado_calibracion = 0
        self.umbral_ear = 0.2
        self.umbral_ear_bajo = 0.2
        self.umbral_ear_cerrado = 0.2
        self.contador_p = 0
        self.suma_frames = 4 #Numero de frames que tiene que estar cerrado el ojo para que se considere un parpadeo
        self.calibrado = False
        self.sonido_click = pygame.mixer.Sound('./sonidos/click.wav')


        # Variables para la recopilacion de datos 
        self.reiniciar_datos_r()
        self.reiniciar_datos_reent()
        self.input = []
        self.output = []
        self.input_frames = []

        # Variable para el modelo
        self.pos_t = (0, 0)
        self.escanear = False

        # Variables para suavizar la mirada en el test
        self.historial = []     # Suaviza la mirada con la mediana esto baja el ruido
        self.historial2 = []    # Suaviza las medianas asi el puntero se mueve suave
        self.cantidad_suavizado = 18
        self.cantidad_suavizado2 = 5
        self.hist_max = 60
        #self.retroceso_click = 0

        # Variables para uso de los tableros
        self.tableros = {}
        self.cargar_tableros()
        self.frase = ""
        self.tablero = None
        self.bloqueado = False
        self.contador_pb = 0
        self.sonido_alarma = pygame.mixer.Sound('./sonidos/alarm.wav')
        self.sonido_lock = pygame.mixer.Sound('./sonidos/lock.wav')
        self.pictogramas = False
        
        #variables para las pruebas de la aplicacion
        self.cronometro_pruebas = 0 #Variable para el cronometro de las pruebas
        self.contador_borrar = 0

        # # Ponderar la mirada
        # self.limiteAbajoIzq = [0.10,0.07]
        # self.limiteAbajoDer = [0.87,0.09]
        # self.limiteArribaIzq = [0.05,0.81]
        # self.limiteArribaDer = [0.93,0.89]
        # self.Desplazamiento = [0.5,0.5]

        # los del optimizador
        # self.limiteAbajoIzq = [0.045,0.048]
        # self.limiteAbajoDer = [0.98,0.006]
        # self.limiteArribaIzq = [0.008,0.94]
        # self.limiteArribaDer = [0.983,0.956]
        # self.Desplazamiento = [0.484,0.45]

        #SIN PONDERAR 
        self.limiteAbajoIzq = [0,0]
        self.limiteAbajoDer = [1,0]
        self.limiteArribaIzq = [0,1]
        self.limiteArribaDer = [1,1]
        self.Desplazamiento = [0.5,0.5]

        # Aplicar un umbral
        self.fondo_frame_editado = cv2.imread('./imagenes/fondo_marco_amp.png', cv2.IMREAD_GRAYSCALE)
        self.mask_rgb = np.zeros((*self.fondo_frame_editado.shape, 3), dtype=np.uint8)
        self.mask_rgb[self.fondo_frame_editado<50] = [50, 50, 50]

        #Variables para el reentrenamiento
        self.numero_entrenamientos = 0


    #Funcion para reiniciar los datos despues de cada escaneo (se aprovecha para inicializarlos tambien)
    def reiniciar_datos_r(self):
        self.recopilar = False #Variable para saber si se esta recopilando datos
        self.contador_r = 5 #Contador para la cuenta atras
        self.pos_r = (0, 0) #Posicion de la pelota roja
        self.salto_bajo, self.salto_alto = 60, 80 #Salto de la pelota roja
        self.velocidad = 25
        self.direccion = 1 

    #Funcion para reiniciar los datos despues de cada reentrenamiento (se aprovecha para inicializarlos tambien)
    #Son menos que en la recopilacion ya que esto es para solamente reajustar al usuario
    def reiniciar_datos_reent(self):
        self.recopilarRe = False
        self.salto_bajo_re, self.salto_alto_re = 100, 180
        self.velocidad_re = 35
        self.numero_epochs = 100
        self.porcentaje_reent = 0
        




    # Cargamos la configuracion del tutorial
    def get_show_tutorial(self):    
        try:
            with open('./config.json', 'r') as f:
                config = json.load(f)
                return config["mostrar_tutorial"]
        except FileNotFoundError:
            return True  # Por defecto, mostrar el tutorial
        
    def set_show_tutorial(self, valor):
        config = {"mostrar_tutorial": valor}
        with open('config.json', 'w') as f:
            json.dump(config, f)

    
    


    
# ---------------------------   FUNCIONES DE CONTROL GENERAL    -------------------------------
#-------------------------------------------------------------------------------------------
        
    def mensaje(self, mensaje):
        # Crear el mensaje de mensaje
        mensaje = Mensajes(mensaje)

        # Agregar el mensaje a la pantalla actual
        App.get_running_app().root.current_screen.add_widget(mensaje)


    def tarea_hilo(self, funcion):
        # Crear un hilo para la tarea
        hilo = threading.Thread(target=funcion)
        hilo.start()

    def salir(self):
        App.get_running_app().stop()

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


    def get_frame_editado(self):
        frame = self.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Poner un texto de que no esta la camara activa alineado al centro y en blanco y con la fuente de texto
            cv2.putText(frame, 'Selecciona una camara', (220, 430), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
            return frame

        # Blanco por defecto
        color = (255, 255, 255)  
        
        # Coger los puntos
        coord_central = self.detector.get_punto_central(frame)
        puntos_or = self.detector.get_puntos_or(frame)

        # Ahora puedes usar mask_rgb en lugar de self.mask

        frame = cv2.addWeighted(frame, 0.5, self.mask_rgb, 1, 0)

        # Crear un circulo en el medio del frame
        r = 10
        # Poner el punto central y la flecha en caso de necesitarla
        if coord_central is not None:
            x = round(frame.shape[1]*coord_central[0])
            y = round(frame.shape[0]*coord_central[1])

            dx = x - frame.shape[1]//2
            dy = y - frame.shape[0]//2

            # Si el punto central esta dentro del circulo de radio r
            if dx**2 + dy**2 < r**2:
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)

                # Calcular el ángulo de la línea desde el centro del círculo hasta el punto central
                angle = math.atan2(dy, dx)

                # Calcular las coordenadas del punto en el borde del círculo
                x_end = frame.shape[1]//2 + r * math.cos(angle)
                y_end = frame.shape[0]//2 + r * math.sin(angle)

                # Dibujar la línea principal de la flecha
                cv2.line(frame, (x, y), (int(x_end), int(y_end)), color, 2)

                # Calcular las coordenadas de los puntos de la punta de la flecha
                arrow_size = np.clip(np.sqrt(dx**2 + dy**2) / 5, 1, 100)
                dx_arrow1 = arrow_size * math.cos(angle + np.pi/4)  # Ajusta el ángulo para la primera línea de la punta de la flecha
                dy_arrow1 = arrow_size * math.sin(angle + np.pi/4)
                dx_arrow2 = arrow_size * math.cos(angle - np.pi/4)  # Ajusta el ángulo para la segunda línea de la punta de la flecha
                dy_arrow2 = arrow_size * math.sin(angle - np.pi/4)

                # Dibujar las líneas de la punta de la flecha
                cv2.line(frame, (int(x_end), int(y_end)), (int(x_end + dx_arrow1), int(y_end + dy_arrow1)), color, 2)
                cv2.line(frame, (int(x_end), int(y_end)), (int(x_end + dx_arrow2), int(y_end + dy_arrow2)), color, 2)

            # Crear un circulo de radio r en el centro
            cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), r, color, 2)

            # Crear un circulo de radio 10 en el punto central
            cv2.circle(frame, (round(frame.shape[1]*coord_central[0]), round(frame.shape[0]*coord_central[1])), 5, color, -1)   

        # Colorear los puntos de la cara
        if puntos_or is not None:
            for punto in puntos_or:
                x = round(frame.shape[1]*punto[0])
                y = round(frame.shape[0]*punto[1])
                frame[y - 1:y + 1, x - 1:x + 1, :] = color


        return frame

# ---------------------------   FUNCIONES CONTROL DEL MENU DE CALIBRACION -------------------------------
#------------------------------------------------------------------------------------------
                

    def obtener_estado_calibracion(self):
        return self.estado_calibracion        
    

# ---------------------------   FUNCIONES DE CONTROL DEL EAR -------------------------------
#-------------------------------------------------------------------------------------------
    
#Para la calibracion
    def cambiar_estado_calibracion(self, numero = None):
        if numero is not None:
            self.estado_calibracion = numero
            return self.estado_calibracion
        # Se obtienen los datos
        datos = self.detector.obtener_coordenadas_indices(self.get_frame())

        #Si no se detecta cara, None
        if datos is None:
            self.mensaje('Calibración fallida, asegúrate de que el usuario está centrado y bien iluminado')
            return None
    
        # Se desempaquetan los datos
        _, _, coord_ear_izq, coord_ear_der, _, _ = self.detector.obtener_coordenadas_indices(self.get_frame())

        # Se captura el EAR actual dependiendo del estado de la calibracion
        if self.estado_calibracion == 1:
            self.umbral_ear_bajo = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        elif self.estado_calibracion == 2:
            self.umbral_ear_cerrado = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)
            self.umbral_ear = (self.umbral_ear_bajo*0.4 + self.umbral_ear_cerrado*0.6) #Se calcula el umbral final ponderado entre el cerrado y el abierto bajo
            self.calibrado = True
        
        elif self.estado_calibracion == 3:
            self.estado_calibracion = 0
            return self.estado_calibracion

        self.estado_calibracion += 1
        return self.estado_calibracion

#Para el test/tableros
    def get_parpadeo(self, ear):
        if ear < self.umbral_ear:
            # Contador para parpadeo
            self.contador_p += 1
            # Contador para el bloqueo de los tableros
            self.contador_pb += 1

            #Si se mantiene cerrado el ojo durante 60 frames, se bloquea el tablero
            if self.contador_pb == 60:  
                self.contador_pb = -1000
                self.bloqueado = not self.bloqueado
                self.sonido_lock.play()
            
            #Si se mantiene cerrado el ojo durante suma_frames, se considera un parpadeo
            if self.contador_p == self.suma_frames:
                self.contador_p = -1000 
                if not self.bloqueado:
                    self.sonido_click.play()
                return 1
            return 0
        else:
            self.contador_p = 0
            self.contador_pb = 0     
            return 0   
    
    def alarma(self):
        self.sonido_alarma.play()

    def set_limite(self, valor, esquina, eje):
        if esquina == 0:
            self.limiteAbajoIzq[eje] = valor
        elif esquina == 1:
            self.limiteAbajoDer[eje] = valor
        elif esquina == 2:
            self.limiteArribaIzq[eje] = valor
        elif esquina == 3:
            self.limiteArribaDer[eje] = valor
        elif esquina == 4:
            self.Desplazamiento[eje] = valor

    def get_limites(self):
        return self.limiteAbajoIzq, self.limiteAbajoDer, self.limiteArribaIzq, self.limiteArribaDer, self.Desplazamiento

# ---------------------------   FUNCIONES DE RECOPILACION DE DATOS  -------------------------------
        
    def cuenta_atras(self, dt):
        if self.contador_r > 0:
            self.contador_r -= 1
        elif self.contador_r == 0:
            Clock.unschedule(self.cuenta_atras)
            return False

    def actualizar_pos_circle_r(self, tamano_pantalla):
        velocidad = self.velocidad_re if self.recopilarRe else self.velocidad
        salto_bajo = self.salto_bajo_re if self.recopilarRe else self.salto_bajo
        salto_alto = self.salto_alto_re if self.recopilarRe else self.salto_alto
    
        # Actualiza la posición x de la pelota
        self.pos_r = (self.pos_r[0] + velocidad * self.direccion, self.pos_r[1])

        # Si la pelota toca los bordes x de la pantalla, cambia la dirección y realiza un salto
        if self.pos_r[0] < 0 or self.pos_r[0] + 50 > tamano_pantalla[0]:
            self.direccion *= -1

            # Actualiza la posición y de la pelota con un salto aleatorio
            salto = random.randint(min(salto_bajo, salto_alto), max(salto_bajo, salto_alto))
            self.pos_r = (self.pos_r[0], self.pos_r[1] + salto)

        # Si la pelota toca el borde superior de la pantalla, invierte el salto
        if self.pos_r[1] + 50 > tamano_pantalla[1]: 
            # Invertimos los saltos y bajamos un poco
            self.salto_bajo, self.salto_alto , self.salto_bajo_re, self.salto_alto_re = self.salto_bajo*-1, self.salto_alto*-1, self.salto_bajo_re*-1, self.salto_alto_re*-1
            self.pos_r = (self.pos_r[0], tamano_pantalla[1] - 50)

        # Si la pelota toca el borde inferior de la pantalla
        if self.pos_r[1] < 0:
            # Si viene de el reentrenamiento, se reentrena 
            if self.recopilarRe:
                self.tarea_hilo(lambda: self.reentrenar())
                self.reiniciar_datos_reent()
            else:
                self.reiniciar_datos_r()
        else:
            datos = self.obtener_datos()
            if datos is not None:                
                distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, frame = datos
                self.guardar_datos(distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, self.pos_r/np.array(tamano_pantalla), frame)
        return self.pos_r


    def guardar_datos(self, distancias_izq, distancias_der, or_x, or_y, or_z, ear, umbral_ear, coord_cab, pos_r_norm, frame):
        # Guardar los datos en las listas
        self.input.append([*distancias_izq, *distancias_der, or_x, or_y, or_z, *coord_cab, ear, umbral_ear])
        self.output.append(pos_r_norm)
        self.input_frames.append(frame)


    def guardar_final(self, fichero):
        def guardar_aux(fichero):
            # Si no existe la carpeta txts, se crea
            os.makedirs('txts', exist_ok=True)
            os.makedirs(f'frames/{fichero}', exist_ok=True)

            # Determinar el número de líneas existentes en el archivo
            with open(f'./entrenamiento/datos/txts/input{fichero}.txt', 'r') as f:
                num_lineas = sum(1 for _ in f)+1

            # Guardar los datos en los archivos
            with open(f'./entrenamiento/datos/txts/input{fichero}.txt', 'a') as f:
                for i, linea in enumerate(self.input):
                    # Convertir el elemento a cadena si es una lista o tupla
                    if isinstance(linea, (list, tuple, np.ndarray)):
                        linea = ', '.join(map(str, linea))
                    f.write(str(linea) + '\n')
                    cv2.imwrite(f'./entrenamiento/datos/frames/{fichero}/frame_{num_lineas}.jpg', self.input_frames[i])
                    num_lineas += 1

            with open(f'./entrenamiento/datos/txts/output{fichero}.txt', 'a') as f:
                for linea in self.output:
                    # Convertir el elemento a cadena si es una lista o tupla
                    if isinstance(linea, (list, tuple, np.ndarray)):
                        linea = ', '.join(map(str, linea))
                    f.write(str(linea) + '\n')

            # Limpiar las listas para la próxima vez
            self.input = []
            self.output = []
            self.input_frames = []
        self.tarea_hilo(lambda: guardar_aux(fichero))

    
    def descartar_datos(self):
        self.input = []
        self.output = []
        self.input_frames = []


       


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
            self.mensaje("No se detecta ninguna cara")
            return None
        
        # Se desempaquetan los datos
        coord_o_izq, coord_o_der, coord_ear_izq, coord_ear_der, coord_cab, coord_o = datos

        # - Distancias de los ojos 
        distancias_izq, distancias_der = self.detector.calcular_distancias_ojos(coord_o_izq, coord_o_der)

        # - Orientación de la cabeza entre 0 y 1
        or_x, or_y, or_z = self.detector.get_orientacion_cabeza(coord_o, frame.shape)

        # - EAR 
        ear = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        # - Recortar el rectangulo de los ojos normalizado a 200x50 ESTO SE HACE EN EL POSTPROCESAR NO AL OBTENER LOS DATOS
        #rect_frame = self.normalizar_frame(frame, coord_o_izq, coord_o_der)

        #print("ORX: ", round(or_x,3), "ORY: ", round(or_y,3), "ORZ: ", round(or_z,3), "coord_cab: ", coord_cab)

        # Pasamos la posicion de la pantalla normalizada
        return distancias_izq, distancias_der, or_x, or_y, or_z, ear, self.umbral_ear, coord_cab, frame

    # Recortar el rectangulo de los ojos normalizado a 200x50
    def normalizar_frame(self, frame, coord_o_izq, coord_o_der):
        # Coordenadas de los ojos
        x_o_izq, y_o_izq = coord_o_izq[0]
        x_o_der, y_o_der = coord_o_der[0]

        # Coordenadas del rectangulo
        x1 = min(x_o_izq, x_o_der)-10
        x2 = max(x_o_izq, x_o_der)+10
        y1 = min(y_o_izq, y_o_der)-10
        y2 = max(y_o_izq, y_o_der)+10

        # Recortar el rectangulo de los ojos
        rect_frame = frame[y1:y2, x1:x2]

        # Calcular la relación de aspecto deseada
        ratio = 200.0 / 50.0
        ratio_act = rect_frame.shape[1] / rect_frame.shape[0]

        # Calcular cuántos píxeles se deben agregar a cada lado
        if ratio_act < ratio:
            diff = int((rect_frame.shape[0] * ratio - rect_frame.shape[1]) / 2)
            rect_frame = cv2.copyMakeBorder(rect_frame, 0, 0, diff, diff, cv2.BORDER_REPLICATE)
        elif ratio_act > ratio:
            diff = int((rect_frame.shape[1] / ratio - rect_frame.shape[0]) / 2)
            rect_frame = cv2.copyMakeBorder(rect_frame, diff, diff, 0, 0, cv2.BORDER_REPLICATE)

        # Redimensionar a 200x50 manteniendo la relación de aspecto
        rect_frame = cv2.resize(rect_frame, (200, 50), interpolation = cv2.INTER_AREA)

        return rect_frame


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
        _, _, _, _, _, ear, _, _, _ = datos
        click = self.get_parpadeo(ear)

        # Se transforman los datos a un conjunto
        datos_array = Conjuntos.datos_as_array(datos)
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

        # Postprocesar la posición de la mirada
        mirada = self.postprocesar(mirada)

        return mirada, click


    def postprocesar(self, mirada):
        # Ponderar la mirada
        mirada = self.ponderar(mirada)

        # Añadir la nueva posición al historial
        self.historial.append(mirada)

        # Eliminar la posición más asntigua si el historial es demasiado largo
        if len(self.historial) > self.hist_max:
            self.historial.pop(0)

        # Primero con la mediana para eliminar ruido y no perder tanto retraso
        mirada = np.median(self.historial[-self.cantidad_suavizado:], axis=0)

        self.historial2.append(mirada)
        if len(self.historial2) > self.hist_max:
            self.historial2.pop(0)
        # Despues con la media de las medianas para suavizar el trazado del puntero
        mirada = np.mean(self.historial2[-self.cantidad_suavizado2:], axis=0)
        
        return mirada


    def ponderar(self, mirada):
        def calcular_limites_esquina(cuadrante):
            if cuadrante == 0:
                return self.limiteAbajoIzq[0], self.limiteAbajoIzq[1], self.limiteAbajoDer[0], self.limiteArribaIzq[1]
            elif cuadrante == 1:
                return self.limiteAbajoIzq[0], self.limiteAbajoDer[1], self.limiteAbajoDer[0], self.limiteArribaDer[1]
            elif cuadrante == 2:
                return self.limiteArribaIzq[0], self.limiteAbajoIzq[1], self.limiteArribaDer[0], self.limiteArribaIzq[1]
            elif cuadrante == 3:
                return self.limiteArribaIzq[0], self.limiteAbajoDer[1], self.limiteArribaDer[0], self.limiteArribaDer[1]
        
        def ponderar_esquina(mirada, esquina_limites):
            LimiteBajoX, LimiteBajoY, LimiteAltoX, LimiteAltoY = esquina_limites

            # Calculamos los límites de la zona no afectada
            ComienzoZonaNoAfectadaX = LimiteBajoX + (self.Desplazamiento[0] - LimiteBajoX) / 2
            FinZonaNoAfectadaX = LimiteAltoX - (LimiteAltoX - self.Desplazamiento[0]) / 2
            ComienzoZonaNoAfectadaY = LimiteBajoY + (self.Desplazamiento[1] - LimiteBajoY) / 2
            FinZonaNoAfectadaY = LimiteAltoY - (LimiteAltoY - self.Desplazamiento[1]) / 2

            # Calculamos las x y las y de las Xs
            Xx = np.array([LimiteBajoX, ComienzoZonaNoAfectadaX, self.Desplazamiento[0], FinZonaNoAfectadaX, LimiteAltoX])
            Xy = np.array([0, ComienzoZonaNoAfectadaX, 0.5, FinZonaNoAfectadaX, 1])
            Yx = np.array([LimiteBajoY, ComienzoZonaNoAfectadaY, self.Desplazamiento[1], FinZonaNoAfectadaY, LimiteAltoY])
            Yy = np.array([0, ComienzoZonaNoAfectadaY, 0.5, FinZonaNoAfectadaY, 1])

            # Crear la función polinómica
            polinomioX = np.poly1d(np.polyfit(Xx, Xy, 4))
            polinomioY = np.poly1d(np.polyfit(Yx, Yy, 4))

            # Calcular el valor ponderado
            return np.array([np.clip(polinomioX(mirada[0]), 0, 1), np.clip(polinomioY(mirada[1]), 0, 1)])

        def calcular_distancia(mirada, esquina):
            return np.sqrt((mirada[0] - esquina[0])**2 + (mirada[1] - esquina[1])**2)

        # Definir las cuatro esquinas
        esquinas = [self.limiteAbajoIzq, self.limiteAbajoDer, self.limiteArribaIzq, self.limiteArribaDer]
        ponderaciones = []

        # Calcular la ponderación para cada esquina
        for esquina_limites in esquinas:
            ponderacion_esquina = ponderar_esquina(mirada, calcular_limites_esquina(esquinas.index(esquina_limites)))
            ponderaciones.append(ponderacion_esquina)

        
        # Calcular la distancia de la mirada a cada esquina
        distancias = [calcular_distancia(mirada, esquina) for esquina in esquinas]

        # Normalizar las distancias para obtener pesos de ponderación
        pesos = np.array([1 / (distancia*2 + 1) for distancia in distancias])  # Cambio aquí

        # Realizar normalizacion min-max de los pesos
        pesos = (pesos - np.min(pesos)) / (np.max(pesos) - np.min(pesos))

        # Sumar los pesos
        suma_pesos = np.sum(pesos)

        # Ponderar las ponderaciones de acuerdo a las distancias
        ponderacion_final = np.zeros_like(ponderaciones[0])
        for i, ponderacion_esquina in enumerate(ponderaciones):
            ponderacion_final += ponderacion_esquina * (pesos[i] / suma_pesos)

        return ponderacion_final

            
#--------------------------------FUNCIONES PARA LOS TABLEROS--------------------------------
#-------------------------------------------------------------------------------------------

    def cargar_tableros(self):
        filename = './tableros/tableros.xlsx'  # Nombre del archivo de Excel
        if os.path.isfile(filename):
            xls = pd.ExcelFile(filename)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                palabras = df.values.tolist()  # Convierte el DataFrame a una lista de listas
                self.tableros[sheet_name] = palabras  # Añade el tablero al diccionario

    def obtener_tablero(self, nombre):
        return self.tableros.get(nombre.lower())
    
    def get_frase(self):
        return self.frase
    
    def añadir_palabra(self, palabra):
        self.frase += palabra + ' '

    def borrar_palabra(self):
        self.frase = ' '.join(self.frase.rstrip().split(' ')[:-1]) + ' '
        self.contador_borrar += 1

    def borrar_todo(self):
        numero_palabras = len(self.frase.split(' '))
        self.contador_borrar += numero_palabras
        self.frase = ''


    def reproducir_texto(self):
        #Empezar un hulo separado:
        def reproducir_texto_hilo():
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

        #Se crea un hilo para reproducir el texto
        self.tarea_hilo(lambda: reproducir_texto_hilo())

    def get_bloqueado(self):
        return self.bloqueado
    
    def set_bloqueado(self, valor):
        self.bloqueado = valor

    def cronometro(self, dt):
        self.cronometro_pruebas += dt

    def iniciar_cronometro(self):
        Clock.schedule_interval(self.cronometro, 0.01)
        self.contador_borrar = 0

    def get_cronometro(self):
        return self.cronometro_pruebas
    
    #Detiene el cronometro y guarda en el txt de resultados de los test el tiempo seguido de la frase
    def stop_cronometro(self):
        Clock.unschedule(self.cronometro)


    def reiniciar_cronometro(self):
        Clock.unschedule(self.cronometro)
        self.cronometro_pruebas = 0
        self.contador_borrar = 0


    
    #----------------------------------- FUNCIONES PARA EL REEENTRENAMIENTO --------------------------------
    #-----------------------------------------------------------------------------------------------------

    def reentrenar(self):
        # Se obtienen los datos
        self.input = np.array(self.input)
        self.output = np.array(self.output)

        # Si los datos estan vacios se pone el porcentaje a 100
        if len(self.input) < 10:
            print("No hay datos para reentrenar")
            self.porcentaje_reent = -1
            self.input = []
            self.output = []
            return

        # Se eliminan los datos con el ojo cerrado
        index = np.where(self.input[:, -2] < self.input[:, -1])
        input = np.delete(self.input, index, axis=0)
        output = np.delete(self.output, index, axis=0)

        # Se obtiene el conjunto 
        normalizar_funcion = getattr(Conjuntos, f'conjunto_{self.conjunto}')
        input = normalizar_funcion(input)

        # Se convierten a tensores
        input_train = torch.from_numpy(input).float()
        output_train = torch.from_numpy(output).float()

        # Se reentrena al modelo
        # Definir el optimizador
        optimizer = optim.Adam(self.modelo.parameters(), lr=0.001)

        train_losses = []
        best_loss = float('inf')
        models = []
        loss = nn.MSELoss()
        first_loss = loss(self.modelo(input_train), output_train).item()
        #COMPROBAR EL NUMERO DE EPOCHS CREO QUE SON DEMASIADOS FAVORECE AL OVERFITTING
        for epoch in range(self.numero_epochs+1):
            # Entrenamiento y cálculo de la pérdida
            train_predictions = self.modelo(input_train)
            train_loss = loss(train_predictions, output_train)
            train_losses.append(train_loss.item())
            print("Epoch: ", epoch, " Loss: ", train_loss.item())

            # Guardar el mejor modelo
            if train_loss.item() < best_loss:
                best_loss = train_loss.item()
            models.append(self.modelo)

            # Actualizar el modelo
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #Guardar el porcentaje de epoch que llevamos
            self.porcentaje_reent = int((epoch/self.numero_epochs)*100)

        self.modelo = models[train_losses.index(min(train_losses))]
        print("Se ha elegido el epoch ", train_losses.index(min(train_losses)))
        print("Perdida antes del reentreno a los valores del usuario: ", first_loss, "Perdida final: ", min(train_losses))
        
        # Se limpian los datos
        self.input = []
        self.output = []


    def descartar_reentrenamientos(self):
        if self.numero_entrenamientos == 0:
            self.mensaje('No hay reentrenamientos que descartar')
            return
        self.modelo = self.modelo_org
        self.mensaje(f'Se han descartado {self.numero_entrenamientos} reentrenamientos')
        self.numero_entrenamientos = 0
