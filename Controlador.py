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

    def get_frame_editado(self, porcentaje):
        return self.modelo.get_frame_editado(porcentaje)
    
    def get_camara_seleccionada(self):
        return self.modelo.get_index_actual()
    

# FUNCIONES PARA EL MENU DE CALIBRACION DEL PARPADEO
    def cambiar_estado_calibracion(self, numero=-1):
        #Primero actualiza el modelo
        n = self.modelo.cambiar_estado_calibracion(numero)

        #Avisa del cambio a la vista para cambiar el texto
        if n == 3:
            self.modelo.cambiar_estado_calibracion(0)
            self.change_screen('inicio')
            self.vista.calibrar.update_view(0)

        else:
            self.vista.calibrar.update_view(n)

    def obtener_estado_cal(self):
        return self.modelo.obtener_estado_calibracion()
    
    def on_continuar_calibracion(self):
        if self.modelo.calibrar_ear() is None:
            return
        self.cambiar_estado_calibracion()
        self.vista.calibrar.update_view(self.obtener_estado_cal())

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

#-------------------------------FUNCIONES PARA LA RECOPILACION DE DATOS-----------------------------

    def on_recopilar(self):
        # Inicia la cuenta atrás
        Clock.schedule_interval(self.modelo.cuenta_atras, 1)

    def get_contador_reco(self):
        # Obtiene el contador del modelo
        return self.modelo.contador_r
    
    
    # Para los textos al recopilar datos
    def set_escaneado(self, valor):
        self.modelo.escaneado = valor

    def get_escaneado(self):
        return self.modelo.escaneado  

    def actualizar_pos_circle_r(self, tamano_pantalla, fichero):
            return self.modelo.actualizar_pos_circle_r(tamano_pantalla, fichero)
    
    def recopilar_datos(self):
        self.modelo.recopilar = True

    def get_recopilando(self):
        return self.modelo.recopilar

    def reiniciar_datos_r(self):
        self.modelo.reiniciar_datos_r() 


# ---------------------------- FUNCIONES PARA EL MODO DE TABLEROS -------------------------
#------------------------------------------------------------------------------------------
    def on_casilla_tab(self, texto_boton):
            if texto_boton.startswith('TAB.'):
                nombre_tablero = texto_boton[5:]
                nuevo_tablero = self.modelo.obtener_tablero(nombre_tablero)
                self.modelo.tablero_actual = nuevo_tablero
                if nuevo_tablero is not None:
                    self.vista.tableros.cambiar_tablero(nuevo_tablero)
            else:
                self.modelo.añadir_palabra(texto_boton)
                self.vista.tableros.label.text = self.modelo.get_frase()
    
    def borrar_palabra(self):
        self.modelo.borrar_palabra()  
        self.vista.tableros.label.text = self.modelo.get_frase()

    def borrar_todo(self):
        self.modelo.borrar_todo()
        self.vista.tableros.label.text = self.modelo.get_frase()

    def obtener_tablero(self, nombre_tablero):
        return self.modelo.obtener_tablero(nombre_tablero)
    
    def get_frase(self):
        return self.modelo.get_frase()


