from kivy.clock import Clock

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
        self.modelo.calibrar_ear(self.modelo.get_frame())
        self.cambiar_estado_calibracion()    
        self.vista.calibrar.update_view(self.obtener_estado_cal())



# ---------------------------- FUNCIONES PARA EL MODO DE TEST -------------------------

# Funcion para reconocer el parpadeo
    def get_parpadeo(self):
        return self.modelo.get_parpadeo()


# Funcion para obtener las distancias de los ojos
    def get_distancias_ojos(self):
        return self.modelo.get_distancias_ojos()
    






#-------------------------------FUNCIONES PARA LA RECOPILACION DE DATOS-----------------------------

    def on_continuar_reco(self):
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

    def reiniciar_datos(self):
        self.modelo.reiniciar_datos_r() 