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
    
    def obtener_camaras(self):
        self.vista.inicio.camera_spinner.text = 'Cargando cámaras...'
        threading.Thread(target=self._obtener_camaras_aux).start()

    def _obtener_camaras_aux(self):
        camaras = self.modelo.obtener_camaras()
        # Programamos la actualización de la interfaz de usuario en el hilo principal
        Clock.schedule_once(lambda dt: self._actualizar_spinner(camaras))

    def _actualizar_spinner(self, camaras):
        self.vista.inicio.camera_spinner.values = [str(i) for i in camaras] + ['Actualizar cámaras']
        self.vista.inicio.camera_spinner.text = 'Seleccionar cámara'

    def seleccionar_camara(self, camara):
        self.modelo.seleccionar_camara(camara)
    

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
        if self.modelo.calibrar_ear() == 1:
            self.mensaje('Calibración fallida, intente de nuevo')
            return
        self.cambiar_estado_calibracion()
        self.vista.calibrar.update_view(self.obtener_estado_cal())



# ---------------------------- FUNCIONES PARA EL MODO DE TEST/TABLEROS -------------------------

# Funcion para reconocer el parpadeo
    def get_escanear(self):
        return self.modelo.escanear
    
    def set_escanear(self, valor):
        self.modelo.escanear = valor

    #tamaño pantalla creo falta
    def obtener_posicion_mirada_ear(self):
        return self.modelo.obtener_posicion_mirada_ear()
        
    




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
    



    def actualizar_pos_circle_r(self, tamano_pantalla):
            return self.modelo.actualizar_pos_circle_r(tamano_pantalla)
    
    def recopilar_datos(self):
        self.modelo.recopilar = True

    def get_recopilando(self):
        return self.modelo.recopilar

    def reiniciar_datos_r(self):
        self.modelo.reiniciar_datos_r() 