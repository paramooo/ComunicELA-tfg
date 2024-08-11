from kivy.clock import Clock
import threading

class Controlador:
    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

    # FUNCIONES PARA EL MANEJO DE PANTALLAS
    def change_screen(self, screen_name):
        self.vista.change_screen(screen_name)    

    #Condicionante para ver si esta calibrado
    def change_screen_r(self, screen_name):
        if self.modelo.calibrado:
            self.change_screen(screen_name)
        else:
            self.mensaje(self.get_string('mensaje_primero_calibrar'))

    #Condicionante para ver si hay camara seleccionada
    def change_screen_cam(self, screen_name):
        if self.modelo.camara_activa():
            self.change_screen(screen_name)
        else:
            self.mensaje(self.get_string('mensaje_primero_camara'))

    def get_screen(self):
        return self.vista.get_screen()
    
    def get_fondo(self):
        return self.vista.get_fondo()
    
    def get_fondo2(self):
        return self.vista.get_fondo2()


# FUNCION PARA EL MAJNEJO DEL IDIOMA
    def get_string(self, nombre):
        return self.modelo.get_string(nombre)
    
    def cambiar_idioma(self):
        self.modelo.cambiar_idioma()
        #self.vista.cambiar_idioma()

    def get_idioma_string(self):
        return self.modelo.get_idioma_string()
    
    def get_idioma_imagen(self):
        return self.modelo.get_idioma_imagen()

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
    
    def get_frame_editado(self):
        return self.modelo.get_frame_editado()
    
    # ---------------------------------------------------------------------------------
    # FUNCIONES PARA EL SPINNER DE CÁMARAS
    # ---------------------------------------------------------------------------------
    def obtener_camaras(self, stop=True):
        self.vista.inicio.camera_spinner.text = self.get_string('cargando_camara')
        threading.Thread(target=self._obtener_camaras_aux, args=[stop]).start()
    

    def _obtener_camaras_aux(self, stop=True):
        camaras = self.modelo.obtener_camaras(stop = stop)
        print(camaras, stop)
        # Programamos la actualización de la interfaz de usuario en el hilo principal
        Clock.schedule_once(lambda dt: self._actualizar_spinner(camaras))

    def _actualizar_spinner(self, camaras):
        self.vista.inicio.camera_spinner.values = [self.get_string('camara_principal') if i == 0 else self.get_string('camara') + str(i) for i in camaras] + [self.get_string('actualizar_camaras')]
        if self.modelo.camara_act is not None:
            self.vista.inicio.camera_spinner.text = self.get_string('camara') + (str(self.modelo.camara_act) if self.modelo.camara_act != 0 else 'principal')
        else:
            self.vista.inicio.camera_spinner.text = self.get_string('btn_inicioDes_seleccionarCam')

    def seleccionar_camara(self, camara):
        self.modelo.seleccionar_camara(camara)
    
    def get_camara_seleccionada(self):
        return self.modelo.get_index_actual()
    
    # def set_camara_seleccionada(self, camara):
    #     self.modelo.sele(camara)
    

#---------------------------------------------------------------------------------
# FUNCIONES PARA EL SPINNER DE VOCES
#---------------------------------------------------------------------------------

    def get_voces(self):    
        voces = self.modelo.get_voces()
        voces.append(self.get_string('actualizar_voces'))
        return voces
    
    def get_voz_seleccionada(self):
        return self.modelo.get_voz_seleccionada()
    
    def seleccionar_voz(self, voz_description):
        self.modelo.seleccionar_voz(voz_description)


#---------------------------------------------------------------------------------
# FUNCIONES PARA EL CORRECTOR DE GEMINI
#---------------------------------------------------------------------------------
    def cambiar_estado_corrector(self):
        return self.modelo.cambiar_estado_corrector()

    def get_estado_corrector(self, mensajes=True):
        return self.modelo.get_corrector_frases(mensajes)

    def internet_available(self):
        return self.modelo.internet_available()
    
    def api_configurada(self):
        return self.modelo.api_bien_configurada()








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
    
    def stop_cronometro(self):
        self.modelo.stop_cronometro()

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

    def actualizar_pos_circle_r(self, tamano_pantalla):
            return self.modelo.actualizar_pos_circle_r(tamano_pantalla)

    def get_recopilando(self):
        return self.modelo.recopilar

    def reiniciar_datos_r(self):
        self.modelo.reiniciar_datos_r() 
    
    #Funciones para el popup de guardar o descartar
    def descartar_datos(self):
        self.modelo.descartar_datos()

    def guardar_final(self, fichero):
        self.modelo.guardar_final(fichero)


# ---------------------------- FUNCIONES PARA EL MODO DE REENTRENAMIENTO -------------------------

    def reiniciar_datos_ree(self):
        self.modelo.reiniciar_datos_reent()

    def set_reentrenando(self, valor):
        self.modelo.recopilarRe = valor

    def get_reent_porcentaje(self):
        return self.modelo.porcentaje_reent
    
    def sumar_reentrenamiento(self):
        self.modelo.numero_entrenamientos += 1
    
    def get_numero_entrenamientos(self):
        return self.modelo.numero_entrenamientos

    def descartar_reentrenamientos(self):
        self.modelo.descartar_reentrenamientos()

    def get_optimizando(self):
        return self.modelo.optimizando
    
    def get_progreso_opt(self):
        return self.modelo.progreso_opt

# ---------------------------- FUNCIONES PARA EL MODO DE TABLEROS -------------------------
#------------------------------------------------------------------------------------------
#TENGO QUE MOVER ESTO AL MODELO Y DEVOLVER EN LA FUNCION LA PALABRA QUE SE HA AÑADIDO 
    def on_casilla_tab(self, texto_boton):
            if texto_boton.startswith('TAB.'):
                nombre_tablero = texto_boton[5:]
                nuevo_tablero = self.modelo.obtener_tablero(nombre_tablero)
                self.modelo.tablero_actual = nuevo_tablero
                if nuevo_tablero is not None:
                    if self.vista.get_screen() == 'tablerosprueb':
                        self.vista.tablerosprueb.cambiar_tablero(nuevo_tablero)
                    else:
                        self.vista.tableros.cambiar_tablero(nuevo_tablero)
            else:
                self.modelo.añadir_palabra(texto_boton)
    
    def borrar_palabra(self):
        self.modelo.borrar_palabra()  

    def borrar_todo(self):
        self.modelo.borrar_todo()

    def obtener_tablero(self, nombre_tablero):
        return self.modelo.obtener_tablero(nombre_tablero)
    
    def obtener_tablero_inicial(self):
        return self.modelo.obtener_tablero_inicial()
    
    def get_frase(self):
        return self.modelo.get_frase()
    
    def get_errores(self):
        return self.modelo.contador_borrar


    def salir(self):
        self.modelo.salir()