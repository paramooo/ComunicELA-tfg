from kivy.clock import Clock
import threading

class Controlador:
    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

# FUNCION GENERAL PARA CAMBIAR ENTRE PANTALLAS
    def change_screen(self, screen_name):
        self.vista.change_screen(screen_name)

    def change_screen_r(self, screen_name):
        if self.modelo.calibrado:
            self.change_screen(screen_name)
        else:
            self.mensaje('Primero debe calibrar el parpadeo')
    def get_screen(self):
        return self.vista.get_screen()
    
    def get_fondo(self):
        return self.vista.get_fondo()
    
    def get_fondo2(self):
        return self.vista.get_fondo2()



# FUNCION GENERAL PARA EL MANEJO DE MENSAJES AL USUARIO
    def mensaje(self, mensaje):
        self.modelo.mensaje(mensaje)

    def get_font_txts(self):
        return self.modelo.tamaño_fuente_txts
    
    def set_desarrollador(self, valor):
        self.modelo.desarrollador = valor

    def get_desarrollador(self):
        return self.modelo.desarrollador
    
    def get_show_tutorial(self):
        return self.modelo.get_show_tutorial()
    
    def set_show_tutorial(self, valor):
        self.modelo.set_show_tutorial(valor)



# FUNCIONES PARA EL MANEJO DE LA CAMARA        
    def iniciar_camara(self):
        self.modelo.iniciar_camara()

    def detener_camara(self):
        self.modelo.detener_camara()

    def camara_activa(self):
        return self.modelo.camara_activa()
    
    def get_frame(self):
        return self.modelo.get_frame()
    
    def obtener_camaras(self):
        self.vista.inicio.camera_spinner.text = 'Cargando cámaras...'
        threading.Thread(target=self._obtener_camaras_aux).start()

    def _obtener_camaras_aux(self):
        camaras = self.modelo.obtener_camaras()
        # Programamos la actualización de la interfaz de usuario en el hilo principal
        Clock.schedule_once(lambda dt: self._actualizar_spinner(camaras))

    def _actualizar_spinner(self, camaras):
        self.vista.inicio.camera_spinner.values = ['Cámara ' + str(i) for i in camaras] + ['Actualizar cámaras']
        self.vista.inicio.camera_spinner.text = 'Seleccionar cámara'

    def seleccionar_camara(self, camara):
        self.modelo.seleccionar_camara(camara)

    def get_frame_editado(self):
        return self.modelo.get_frame_editado()
    
    def get_camara_seleccionada(self):
        return self.modelo.get_index_actual()
    

# FUNCIONES PARA EL MENU DE CALIBRACION DEL PARPADEO    
    def obtener_estado_cal(self):
        return self.modelo.obtener_estado_calibracion()
    
    def cambiar_estado_calibracion(self, numero=None):#MOVER ESTO AL MODELO
        return self.modelo.cambiar_estado_calibracion(numero)
    
    def get_punto_central(self, frame):
        return self.modelo.get_punto_central(frame)
    

# ---------------------------- FUNCIONES PARA EL MODO DE TEST/TABLEROS -------------------------

# Funcion para reconocer el parpadeo
    def get_escanear(self):
        return self.modelo.escanear
    
    def set_escanear(self, valor):
        self.modelo.escanear = valor

    #tamaño pantalla creo falta
    def obtener_posicion_mirada_ear(self):
        return self.modelo.obtener_posicion_mirada_ear()
    
    def reproducir_texto(self):
        self.modelo.reproducir_texto()

    def set_limite(self, valor, esquina, eje):
        self.modelo.set_limite(valor, esquina, eje)

    def get_limites(self):
        return self.modelo.get_limites()
    
    def get_bloqueado(self):
        return self.modelo.bloqueado
    
    def set_bloqueado(self, valor):
        self.modelo.bloqueado = valor

    def reproducir_alarma(self):
        self.modelo.alarma()

    def iniciar_cronometro(self):
        self.modelo.iniciar_cronometro()

    def reiniciar_cronometro(self):
        self.modelo.reiniciar_cronometro()

    def get_cronometro (self):
        return self.modelo.get_cronometro()
    
    def stop_cronometro(self, guardar):
        self.modelo.stop_cronometro(guardar)

    def get_pictogramas(self):
        return self.modelo.pictogramas

    def set_pictogramas(self, valor):
        self.modelo.pictogramas = valor

#-------------------------------FUNCIONES PARA LA RECOPILACION DE DATOS-----------------------------

    def on_recopilar(self):
        # Inicia la cuenta atrás
        Clock.schedule_interval(self.modelo.cuenta_atras, 1)
        self.modelo.recopilar = True

    def get_contador_reco(self):
        # Obtiene el contador del modelo
        return self.modelo.contador_r
    
    
    # Para los textos al recopilar datos
    def set_escaneado(self, valor):
        self.modelo.escaneado = valor

    def get_escaneado(self):
        return self.modelo.escaneado  

    def actualizar_pos_circle_r(self, tamano_pantalla, fichero=None):
            return self.modelo.actualizar_pos_circle_r(tamano_pantalla, fichero)

    def get_recopilando(self):
        return self.modelo.recopilar

    def reiniciar_datos_r(self):
        self.modelo.reiniciar_datos_r() 


# ---------------------------- FUNCIONES PARA EL MODO DE REENTRENAMIENTO -------------------------

    def reiniciar_datos_ree(self):
        self.modelo.reiniciar_datos_reent()

    def set_reentrenando(self, valor):
        self.modelo.recopilarRe = valor

    def mostrar_reentrenando(self):
        self.vista.reentrenar.texto_explicativo.text = 'Reentrenando...'

# ---------------------------- FUNCIONES PARA EL MODO DE TABLEROS -------------------------
#------------------------------------------------------------------------------------------
#TENGO QUE MOVER ESTO AL MODELO Y DEVOLVER EN LA FUNCION LA PALABRA QUE SE HA AÑADIDO 
    def on_casilla_tab(self, texto_boton):
            if texto_boton.startswith('TAB.'):
                nombre_tablero = texto_boton[5:]
                nuevo_tablero = self.modelo.obtener_tablero(nombre_tablero)
                self.modelo.tablero_actual = nuevo_tablero
                if nuevo_tablero is not None:
                    self.vista.tableros.cambiar_tablero(nuevo_tablero)
            else:
                self.modelo.añadir_palabra(texto_boton)
    
    def borrar_palabra(self):
        self.modelo.borrar_palabra()  

    def borrar_todo(self):
        self.modelo.borrar_todo()

    def obtener_tablero(self, nombre_tablero):
        return self.modelo.obtener_tablero(nombre_tablero)
    
    def get_frase(self):
        return self.modelo.get_frase()


