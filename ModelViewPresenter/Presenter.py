from kivy.clock import Clock
from threading import Thread

class Presenter:
    """
    Clase Presenter del patrón MVP que se encarga de la comunicación entre el modelo y la vista
    """

    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

    # ---------------------------- FUNCIONES PARA EL MANEJO DE PANTALLAS -------------------------
    # --------------------------------------------------------------------------------------------
    def change_screen(self, screen_name):
        """
        Cambia la pantalla actual de la aplicación

        Args:
            screen_name (str): Nombre de la pantalla a la que se quiere cambiar
        """
        self.vista.change_screen(screen_name)    



    def change_screen_r(self, screen_name):
        """
        Comprueba antes de cambiar que se haya calibrado antes

        Args:
            screen_name (str): Nombre de la pantalla a la que se quiere cambiar
        """
        if self.modelo.camara_activa() and self.modelo.calibrado:
            self.change_screen(screen_name)
        else:
            self.mensaje(self.get_string('mensaje_primero_calibrar'))



    def change_screen_cam(self, screen_name):
        """
        Comprueba antes de cambiar que se haya activado la cámara

        Args:
            screen_name (str): Nombre de la pantalla a la que se quiere cambiar
        """
        if self.modelo.camara_activa():
            self.change_screen(screen_name)
        else:
            self.mensaje(self.get_string('mensaje_primero_camara'))



    def get_screen(self):
        """
        Devuelve la pantalla actual de la aplicación
        """
        return self.vista.get_screen()
    


    def get_fondo(self):
        """
        Devuelve el fondo de pantalla 1
        """
        return self.vista.get_fondo()
    


    def get_fondo2(self):
        """
        Devuelve el fondo de pantalla 2
        """
        return self.vista.get_fondo2()



# ------------------------------------- FUNCION PARA EL MAJNEJO DEL IDIOMA------------------------
# ------------------------------------------------------------------------------------------------

    def get_string(self, nombre):
        """
        Comunica con el modelo para obtener un string del idioma actual

        Args:
            nombre (str): Nombre del string a obtener

        Returns:
            str: String del idioma actual
        """
        return self.modelo.get_string(nombre)
    


    def cambiar_idioma(self):
        """
        Comunica con el modelo para cambiar el idioma
        """
        self.modelo.cambiar_idioma()



    def get_idioma_string(self):
        """
        Comunica con el modelo para obtener el idioma actual
        """
        return self.modelo.get_idioma_string()
    


    def get_idioma_imagen(self):
        """
        Comunica con el modelo para obtener la imagen del idioma actual
        """
        return self.modelo.get_idioma_imagen()


# ------------------------------------- FUNCIONES PARA EL MANEJO GENERAL ------------------------
# ------------------------------------------------------------------------------------------------

    def mensaje(self, mensaje):
        """
        Comunica con el modelo para mostrar un mensaje informativo

        Args:
            mensaje (str): Mensaje a mostrar
        
        """
        self.modelo.mensaje(mensaje)


    def get_font_txts(self):
        """"
        Obtiene el tamaño de la fuente de los textos
        """
        return self.modelo.tamaño_fuente_txts
    

    def set_desarrollador(self, valor):
        """
        Comunica con el modelo para cambiar la visibilidad de las opciones de desarrollador

        Args:
            valor (bool): Visibilidad de las opciones de desarrollador
        """
        self.modelo.desarrollador = valor


    def get_desarrollador(self):
        """
        Comunica con el modelo para obtener la visibilidad de las opciones de desarrollador
        """
        return self.modelo.desarrollador
    

    def get_show_tutorial(self):
        """
        Comunica con el modelo para obtener si se muestra el tutorial
        """
        return self.modelo.get_show_tutorial()
    

    def set_show_tutorial(self, valor):
        """
        Comunica con el modelo para cambiar si se muestra el tutorial

        Args:
            valor (bool): Si se muestra el tutorial
        """
        self.modelo.set_show_tutorial(valor)



# -------------------------- FUNCIONES PARA EL MANEJO DE LA CAMARA ------------------------------
# -----------------------------------------------------------------------------------------------

    def iniciar_camara(self):
        """
        Comunica con el modelo para iniciar la cámara
        """
        self.modelo.iniciar_camara()


    def detener_camara(self):
        """
        Comunica con el modelo para detener la cámara
        """
        self.modelo.detener_camara()


    def camara_activa(self):
        """
        Comunica con el modelo para saber si la cámara está activa
        """
        return self.modelo.camara_activa()
    

    def get_frame(self):
        """
        Obtiene el frame de la cámara
        """
        return self.modelo.get_frame()
    
    def get_frame_editado(self):
        """
        Obtiene el frame de la cámara editado con la silueta y los puntos
        """
        return self.modelo.get_frame_editado()
    

    # ---------------------- FUNCIONES PARA EL SPINNER DE CÁMARAS ---------------------
    # ---------------------------------------------------------------------------------
    def obtener_camaras(self, stop=True):
        """
        Obtiene las cámaras disponibles y las muestra en el spinner

        Args:
            stop (bool): Indica si se debe detener la cámara
        """
        self.vista.inicio.camera_spinner.text = self.get_string('cargando_camara')
        Thread(target=self._obtener_camaras_aux, args=[stop]).start()
    

    def _obtener_camaras_aux(self, stop=True):
        """
        Funcion auxiliar para obtener las cámaras en un hilo secundario

        Args:
            stop (bool): Indica si se debe detener la cámara
        """
        camaras = self.modelo.obtener_camaras(stop = stop)
        # Programamos la actualización de la interfaz de usuario en el hilo principal
        Clock.schedule_once(lambda dt: self._actualizar_spinner(camaras))


    def _actualizar_spinner(self, camaras):
        """
        Actualiza el spinner de cámaras con las cámaras disponibles

        Args:
            camaras (list): Lista de cámaras disponibles
        """
        self.vista.inicio.camera_spinner.values = [self.get_string('camara_principal') if i == 0 else self.get_string('camara') + str(i) for i in camaras] + [self.get_string('actualizar_camaras')]
        if self.modelo.camara_act is not None:
            self.vista.inicio.camera_spinner.text = self.get_string('camara') + (str(self.modelo.camara_act) if self.modelo.camara_act != 0 else 'principal')
        else:
            self.vista.inicio.camera_spinner.text = self.get_string('btn_inicioDes_seleccionarCam')


    def seleccionar_camara(self, camara):
        """
        Selecciona la cámara indicada

        Args:
            camara (int): Número de la cámara a seleccionar
        """
        self.modelo.seleccionar_camara(camara)
    

    def get_camara_seleccionada(self):
        """
        Obtiene la cámara seleccionada
        """
        return self.modelo.get_index_actual()
    


# ----------------------- FUNCIONES PARA EL SPINNER DE VOCES ---------------------
#---------------------------------------------------------------------------------

    def get_voces(self):  
        """
        Obtiene las voces disponibles y las muestra en el spinner
        """
        voces = self.modelo.get_voces()
        voces.append(self.get_string('actualizar_voces'))
        return voces
    

    def get_voz_seleccionada(self):
        """
        Obtiene la voz seleccionada
        """
        return self.modelo.get_voz_seleccionada()
    

    def seleccionar_voz(self, voz_description):
        """
        Seleciona la voz indicada

        Args:
            voz_description (str): Descripción de la voz a seleccionar
        """
        self.modelo.seleccionar_voz(voz_description)




#---------------------------------------------------------------------------------
# FUNCIONES PARA EL CORRECTOR DE GEMINI
#---------------------------------------------------------------------------------
    def cambiar_estado_corrector(self):
        """
        Cambia el estado del corrector de frases
        """
        return self.modelo.cambiar_estado_corrector()


    def get_estado_corrector(self, mensajes=True):
        """
        Obtiene el estado del corrector de frases
        """
        return self.modelo.get_corrector_frases(mensajes)


    def internet_available(self):
        """
        Comprueba si hay conexión a internet
        """
        return self.modelo.internet_available()
    

    def api_configurada(self):
        """
        Comprueba que la API de Google esté bien configurada
        """
        return self.modelo.api_bien_configurada()





# FUNCIONES PARA EL MENU DE CALIBRACION DEL PARPADEO    
    def obtener_estado_cal(self):
        """
        LLama al modelo para obtener el estado de la calibración
        """
        return self.modelo.obtener_estado_calibracion()
    

    def cambiar_estado_calibracion(self, numero=None):
        """
        LLama al modelo para cambiar el estado de la calibración
        """
        return self.modelo.cambiar_estado_calibracion(numero)
    
    

# ---------------------------- FUNCIONES PARA EL MODO DE TEST/TABLEROS -------------------------
# ----------------------------------------------------------------------------------------------

    def get_escanear(self):
        """
        Obtiene del modelo si se está escaneando
        """
        return self.modelo.escanear
    

    def set_escanear(self, valor):
        """
        Cambia el valor de escaneando en el modelo

        Args:
            valor (bool): Valor a cambiar
        """
        self.modelo.escanear = valor


    def obtener_posicion_mirada_ear(self):
        """
        Obtiene la posición de la mirada del modelo
        """
        return self.modelo.obtener_posicion_mirada_ear()
    

    def reproducir_texto(self):
        """
        LLama al modelo para reproducir el texto
        """
        self.modelo.reproducir_texto()


    def set_limite(self, valor, esquina, eje):
        """
        LLama al modelo para cambiar los limites de la ponderación

        Args:
            valor (float): Valor a cambiar
            esquina (str): Esquina a cambiar
            eje (str): Eje a cambiar
        """
        self.modelo.set_limite(valor, esquina, eje)


    def get_limites(self):
        """
        Obtiene los limites de la ponderación
        """
        return self.modelo.get_limites()
    

    def get_bloqueado(self):
        """
        Obtiene del modelo si la pantalla esta bloqueada
        """
        return self.modelo.bloqueado
    

    def set_bloqueado(self, valor):
        """
        Cambia el valor de bloqueado en el modelo

        Args:
            valor (bool): Valor a cambiar    
        """
        self.modelo.bloqueado = valor


    def reproducir_alarma(self):
        """
        LLama al modelo para reproducir la alarma
        """
        self.modelo.alarma()


    def iniciar_cronometro(self):
        """
        LLama al modelo para iniciar el cronometro
        """
        self.modelo.iniciar_cronometro()


    def reiniciar_cronometro(self):
        """
        LLama al modelo para reiniciar el cronometro
        """
        self.modelo.reiniciar_cronometro()


    def get_cronometro (self):
        """
        Obtiene el valor del cronometro
        """
        return self.modelo.cronometro_pruebas
    

    def stop_cronometro(self):
        """
        Para el cronómetro
        """
        self.modelo.stop_cronometro()


    def get_pictogramas(self):
        """
        Obtiene si los pictogramas están activos o no
        """
        return self.modelo.pictogramas


    def set_pictogramas(self, valor):
        """
        Establece si los pictogramas están activos o no
        """
        self.modelo.pictogramas = valor



#-------------------------------FUNCIONES PARA LA RECOPILACION DE DATOS-----------------------------
#---------------------------------------------------------------------------------------------------
    def on_recopilar(self):
        """
        Se ejecuta al pulsar sobre el boton de recopilar datos
        """
        Clock.schedule_interval(self.modelo.cuenta_atras, 1)
        self.modelo.recopilar = True


    def get_contador_reco(self):
        """
        Obtiene el contador del modelo
        """
        return self.modelo.contador_r
    

    def set_escaneado(self, valor):
        """
        Establece si se ha escaneado ya o no

        Args:
            valor (bool): Valor a cambiar
        """
        self.modelo.escaneado = valor


    def get_escaneado(self):
        """
        Obtiene si se ha escaneado ya o no
        """
        return self.modelo.escaneado  


    def actualizar_pos_circle_r(self, tamano_pantalla):
        """
        Obtiene la posición del circulo de recopilación del modelo

        Args:
            tamano_pantalla (tuple): Tamaño de la pantalla
        """
        return self.modelo.actualizar_pos_circle_r(tamano_pantalla)


    def get_recopilando(self):
        """
        Obtiene del modelo si se está recopilando
        """
        return self.modelo.recopilar


    def reiniciar_datos_r(self):
        """
        LLama al modelo para reiniciar los datos de recopilación
        """
        self.modelo.reiniciar_datos_r() 
    
    def descartar_datos(self):
        """
        Descartar los datos de recopilación
        """
        self.modelo.descartar_datos()

    def guardar_final(self):
        """
        Guarda los datos de la recopilación en los archivos de la bd
        """
        self.modelo.guardar_final()


# ---------------------------- FUNCIONES PARA EL MODO DE REENTRENAMIENTO -------------------------
# ------------------------------------------------------------------------------------------------

    def reiniciar_datos_ree(self):
        """
        Reinicia los datos del reentrenamiento
        """
        self.modelo.reiniciar_datos_reent()


    def set_reentrenando(self, valor):
        """
        Establece si se está reentrenando o no

        Args:
            valor (bool): Valor a cambiar
        """
        self.modelo.recopilarRe = valor


    def get_reent_porcentaje(self):
        """
        Obtiene del modelo el porcentaje de reentrenamiento realizado hasta el momento
        """
        return self.modelo.porcentaje_reent
    

    def sumar_reentrenamiento(self):
        """
        Suma un reentrenamiento al modelo
        """
        self.modelo.numero_entrenamientos += 1
    

    def get_numero_entrenamientos(self):
        """
        Obtiene del modelo el número de reentrenamientos realizados
        """
        return self.modelo.numero_entrenamientos


    def descartar_reentrenamientos(self):
        """
        Descarta los reentrenamientos realizados
        """
        self.modelo.descartar_reentrenamientos()



    def get_optimizando(self):
        """
        Obtiene del modelo si se está optimizando o no
        """
        return self.modelo.optimizando
    


    def get_progreso_opt(self):
        """
        Obtiene del modelo el progreso de la optimización realizado hasta el momento
        """
        return self.modelo.progreso_opt

# ---------------------------- FUNCIONES PARA EL MODO DE TABLEROS -------------------------
#------------------------------------------------------------------------------------------
    def cargar_tableros(self):
        """
        LLama al modelo para cargar los tableros
        """
        return self.modelo.cargar_tableros()


    def on_casilla_tab(self, texto_boton):
        """
        Se ejecuta al pulsar sobre una casilla del tablero, añade la palabra a la frase

        Args:
            texto_boton (str): Texto del botón pulsado
        
        """
        if texto_boton.startswith('TAB.'):
            nuevo_tablero = self.modelo.obtener_tablero(texto_boton)
            self.modelo.tablero_actual = nuevo_tablero
            if nuevo_tablero is not None:
                if self.vista.get_screen() == 'tablerosprueb':
                    self.vista.tablerosprueb.cambiar_tablero(nuevo_tablero)
                else:
                    self.vista.tableros.cambiar_tablero(nuevo_tablero)
        else:
            self.modelo.añadir_palabra(texto_boton)
    
    def borrar_palabra(self):
        """
        LLama al modelo para borrar la última palabra de la frase
        """
        self.modelo.borrar_palabra()  

    def borrar_todo(self):
        """
        LLama al modelo para borrar toda la frase
        """
        self.modelo.borrar_todo()

    def obtener_tablero(self, nombre_tablero):
        """
        Obtiene el tablero indicado

        Args:
            nombre_tablero (str): Nombre del tablero a obtener
        """
        return self.modelo.obtener_tablero(nombre_tablero)
    
    def obtener_tablero_inicial(self):
        """
        Obtiene el tablero inicial
        """
        return self.modelo.obtener_tablero_inicial()
    
    def get_frase(self):
        """
        Obtiene la frase actual
        """
        return self.modelo.frase
    
    def get_errores(self):
        """ 
        Obtiene el número de errores cometidos
        """
        return self.modelo.contador_borrar


    def salir(self):
        """
        Cierra la aplicación
        """
        self.modelo.salir()