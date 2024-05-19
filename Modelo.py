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
import threading
import math


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
        self.modelo = torch.load('./anns/pytorch/modelo_ajustado.pth')

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
        
        # Ponderar la mirada
        self.limiteAbajoIzq = [0.10,0.07]
        self.limiteAbajoDer = [0.87,0.09]
        self.limiteArribaIzq = [0.05,0.81]
        self.limiteArribaDer = [0.93,0.89]
        self.Desplazamiento = [0.5,0.5]

        # Aplicar un umbral
        self.fondo_frame_editado = cv2.imread('./imagenes/fondo_marco_amp.png', cv2.IMREAD_GRAYSCALE)
        self.mask_rgb = np.zeros((*self.fondo_frame_editado.shape, 3), dtype=np.uint8)
        self.mask_rgb[self.fondo_frame_editado<50] = [40, 40, 40]

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

        # Blanco por defecto
        color = (255, 255, 255)  
        
        # Coger los puntos
        coord_central = self.detector.get_punto_central(frame)
        puntos_or = self.detector.get_puntos_or(frame)

        # Ahora puedes usar mask_rgb en lugar de self.mask

        frame = cv2.addWeighted(frame, 1, self.mask_rgb, 0.6, 0)

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

                # Poner una flecha al punto más cercano del círculo de radio r
                cv2.arrowedLine(frame, (x, y), (int(x_end), int(y_end)), color, 2)

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





# ---------------------------   FUNCIONES DE CONTROL DEL EAR -------------------------------
#-------------------------------------------------------------------------------------------
    
#Para la calibracion
    def calibrar_ear(self):
        #Si estamos en el estado 2, no se hace nada, sale al inicio
        if self.estado_calibracion == 2:
            return 0

        # Se obtienen los datos
        datos = self.detector.obtener_coordenadas_indices(self.get_frame())

        #Si no se detecta cara, se devuelve 1 indicando error
        if datos is None:
            self.mensaje('Calibración fallida, intente de nuevo')
            return None
    
        # Se desempaquetan los datos
        _, _, coord_ear_izq, coord_ear_der, _, _ = self.detector.obtener_coordenadas_indices(self.get_frame())

        # Se captura el EAR actual dependiendo del estado de la calibracion
        if self.estado_calibracion == 0:
            self.umbral_ear_bajo = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)

        elif self.estado_calibracion == 1:
            self.umbral_ear_cerrado = self.detector.calcular_ear_medio(coord_ear_izq, coord_ear_der)
            self.umbral_ear = (self.umbral_ear_bajo*0.4 + self.umbral_ear_cerrado*0.6) #Se calcula el umbral final ponderado entre el cerrado y el abierto bajo
            self.calibrado = True
        return 0

#Para el test/tableros
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

        # Imprimir el peso de cada esquina
        print(f'0: {round(pesos[0],6)} \t1: {round(pesos[1], 6)} \t2: {round(pesos[2], 6)} \t3: {round(pesos[3], 6)}')
        return ponderacion_final

            
#--------------------------------FUNCIONES PARA LOS TABLEROS--------------------------------
#-------------------------------------------------------------------------------------------

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
        threading.Thread(target=reproducir_texto_hilo).start()


    