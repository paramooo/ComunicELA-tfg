from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from KivyCustom.Custom import ButtonRnd, CustomTextInput
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics import Color, Line
from KivyCustom.Tablero import Tablero
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from KivyCustom.PopUp import CustPopup
from Vistas.Tableros import PantallaBloqueada
from csv import writer as csv_writer
from ajustes.utils import get_recurso


class TablerosPruebas(Screen):
    """
    Pantalla de las pruebas automatizadas de ComunicELA.
    """
    def __init__(self, controlador, **kwargs):
        super(TablerosPruebas, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1)
        

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo()  )
        self.add_widget(self.fondo)

        # Layout principal
        self.layout_principal = BoxLayout(orientation='vertical')
        self.add_widget(self.layout_principal)

        # Tablero
        self.tablero = None
        self.cambiar_tablero(self.controlador.obtener_tablero('TAB. INICIAL'))

        # Añade un espacio en blanco
        espacio_blanco = BoxLayout(size_hint=(1, .05))

        # Layout de los botones
        layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, .15), spacing=10)
        self.layout_vertical = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        self.layout_vertical.add_widget(espacio_blanco)
        self.layout_vertical.add_widget(layout_botones)
        self.layout_principal.add_widget(self.layout_vertical)

        # El botón de inicio
        self.btn_inicio = ButtonRnd(text='Inicio', size_hint=(.12, 1), on_press=self.on_inicio, font_name='Texto')
        layout_botones.add_widget(self.btn_inicio)

        # Espacio para texto
        scroll = ScrollView(size_hint=(.4, 1), scroll_type=['bars', 'content'], bar_width=10)
        self.label = CustomTextInput(
            text=self.controlador.get_frase(),
            # Limita el ancho del texto al ancho del widget
            #size_hint_y=None,  # Esto permitirá que el TextInput se expanda a su tamaño natural
            height=Window.height * 0.2,  # Altura inicial del TextInput
            halign='left',  
            font_name='Texto', 
            font_size=40,
            background_color=(0.7, 0.7, 0.7, 1),
            foreground_color=(0, 0, 0, 1),
            readonly=True,
            cursor_blink=False,  # Deshabilita el parpadeo del cursor
            cursor_width=0,  # Establece el ancho del cursor a 0
            focus=False
        )
        self.label.bind(on_text=self.on_text)
        self.label.bind(on_touch_down = self.on_label_touch_down)
        scroll.add_widget(self.label)
        layout_botones.add_widget(scroll)

        # El botón para borrar una palabra
        self.btn_borrar_palabra = ButtonRnd(text='Borrar', size_hint=(.15, 1), on_press=self.on_borrar_palabra, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_palabra)

        # El botón para borrar todo el texto
        self.btn_borrar_todo = ButtonRnd(text='Borrar todo', size_hint=(.15, 1), on_press=self.on_borrar_todo, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_todo)

        # El botón para reproducir el texto
        self.btn_reproducir = ButtonRnd(text='Reproducir', size_hint=(.15, 1), on_press=self.on_reproducir, font_name='Texto')
        #layout_botones.add_widget(self.btn_reproducir)

        # El botón de alarma
        self.btn_alarma = ButtonRnd(text='Alarma', size_hint=(.15, 1), on_press=self.on_alarma, font_name='Texto')
        layout_botones.add_widget(self.btn_alarma)

        # Variables para emular el movimiento y clic
        self.casilla_bloqueada = None
        self.contador_frames = 0
        self.casilla_anterior = None
        self.frames_bloqueo = 30
        self.botones = [self.btn_inicio, self.btn_borrar_palabra, self.btn_borrar_todo, self.btn_reproducir, self.btn_alarma]
        self.dibujos_mirada = []

        # Pruebas
        self.pruebas = ["Explicacion", "SÍ", "NO", "BIEN", "MAL", "QUERER BEBER", "PIERNA DERECHA", "ABRIR VENTANA", "YO PODER DORMIR AHORA ?", "YO NECESITAR VER DOCTOR", "EL-ELLA ANTES VIAJAR MUCHO", "NOSOTROS-AS PODER IR PARQUE ?"]
        self.cursor_positions = []
        self.indice_prueba = 0
        self.pruebas_coordenadas = {}
        self.palabras_y_coordenadas = []  # Lista de tuplas (palabra, coordenadas, tiempo, indice_casilla)


    def on_text(self, instance, value):
        """
        Mantiene la vista en la parte superior del texto.
        """
        instance.scroll_y = 0

    def on_pre_enter(self, *args):
        """
        Antes de entrar en la pantalla, se cargan los tableros y se establecen las configuraciones iniciales.
        """
        self.controlador.cargar_tableros()
        self.controlador.set_pictogramas(False)
        #Las primeras pruebas son dentro del tablero rapido (pruebas de seleccionar una palabra solo)
        self.cambiar_tablero(self.controlador.obtener_tablero('TAB. RÁPIDO'))

    def on_enter(self, *args):
        """
        Al entrar en la pantalla, se inician las pruebas.
        """
        self.pruebas_mensajes()
        

    def on_label_touch_down(self, instance, touch):
        """
        Al tocar el texto, se reproduce el audio.
        """
        self.on_reproducir(instance)


    def establecer_configuracion(self):
        """
        Establece la configuración para cada prueba.
        """
        # Pruebas NIVEL BASICO sin pictogramas y en el tablero rapido
        if self.indice_prueba < 5:
            self.controlador.set_pictogramas(False)
            self.cambiar_tablero(self.controlador.obtener_tablero('TAB. RÁPIDO'))
        
        # Pruebas NIVEL INTERMEDIO con pictogramas y en el tablero inicial
        if self.indice_prueba >= 5 and self.indice_prueba < 8:
            self.controlador.set_pictogramas(True)
            self.cambiar_tablero(self.controlador.obtener_tablero('TAB. INICIAL'))
        
        # Pruebas NIVEL AVANZADO con pictogramas y en el tablero inicial
        if self.indice_prueba >= 8 and self.indice_prueba < 10:
            self.controlador.set_pictogramas(True)
            self.cambiar_tablero(self.controlador.obtener_tablero('TAB. INICIAL'))
        
        # Pruebas NIVEL EXPERTO sin pictogramas y en el tablero inicial
        if self.indice_prueba >= 10:
            self.controlador.set_pictogramas(False)
            self.cambiar_tablero(self.controlador.obtener_tablero('TAB. INICIAL'))

    def pruebas_mensajes(self, *args):
        """
        Establece los mensajes para cada prueba.
        """
        if self.pruebas:
            #Establecemos el tablero y las configuraciones necesarias
            self.establecer_configuracion()

            # Mensajes de la explicacion
            if self.indice_prueba == 0:
                message = "Gracias por participar en las pruebas de ComunicELA\nSe pedirá escribir una serie de frases para evaluar\nel tiempo de respuesta y la precisión del software."
                CustPopup(message, self.pruebas_indices, (0.5, 0.5), self.controlador, bt_empez=True, show_switch=False).open()

            # Mensaje de la primera prueba de selección de palabras
            elif self.indice_prueba == 1:
                message = f"Prueba {self.indice_prueba}: Escriba '{self.pruebas[self.indice_prueba]}' y presione 'Reproducir'."
                self.pruebas_coordenadas[self.pruebas[self.indice_prueba]] = {}
                CustPopup(message, self.pruebas_indices, (0.5, 0.5), self.controlador, show_switch=False, bt_empez=True, func_saltar=self.saltar_prueba).open()

            # Mensaje de las pruebas de selección de palabras
            elif self.indice_prueba < len(self.pruebas):
                # Mensaje de las pruebas de selección de palabras
                message = f"¡Prueba completada con éxito!\nPrueba {self.indice_prueba}: Escriba '{self.pruebas[self.indice_prueba]}' y presione 'Reproducir'."

                #Cambio a pictogramas
                if self.indice_prueba == 5:
                    message = f"¡Pruebas anteriores completadas con éxito!\nActivamos los pictogramas y los demás tableros\nPrueba {self.indice_prueba}: Escriba '{self.pruebas[self.indice_prueba]}' y presione 'Reproducir'."

                # Vuelta a texto
                if self.indice_prueba == 10:
                    message = f"¡Pruebas anteriores completadas con éxito!\nDesactivamos los pictogramas de nuevo\nPrueba {self.indice_prueba}: Escriba '{self.pruebas[self.indice_prueba]}' y presione 'Reproducir'."

                self.pruebas_coordenadas[self.pruebas[self.indice_prueba]] = {}
                CustPopup(message, self.pruebas_indices, (0.5, 0.5), self.controlador, show_switch=False, bt_empez=True, func_saltar=self.saltar_prueba, func_volver=self.volver_anterior).open()

            else:
                # Mensaje de fin de pruebas
                message = "¡Gracias! ¡Todas las pruebas han sido completadas!\nPresiona continuar para volver al inicio."
                CustPopup(message, self.on_inicio, (0.5, 0.5), self.controlador, show_switch=False, func_volver=self.volver_anterior).open()



    def pruebas_indices(self, *args):
        """
        Establece los índices de las pruebas.
        """
        self.controlador.borrar_todo()
        if self.indice_prueba == 0:
            self.indice_prueba += 1
            self.pruebas_mensajes()
        else:
            self.start_test()
    

    def saltar_prueba(self, *args):
        """
        Salta la prueba actual.
        """
        self.indice_prueba += 1
        self.pruebas_mensajes()
    

    def volver_anterior(self, *args):
        """
        Vuelve a la prueba anterior.
        """
        self.indice_prueba -= 1
        self.pruebas_mensajes()


    def start_test(self):
        """
        Se inicia cuando empieza la prueba, inicia el escaneo y el cronometro.
        """
        self.controlador.set_escanear(True)
        self.controlador.set_bloqueado(False)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        self.controlador.reiniciar_cronometro()
        self.controlador.iniciar_cronometro()

    def on_inicio(self, *args):
        """
        Al tocar sobre el boton de inicio, se cambia a la pantalla de inicio y se limpian los datos de la pantalla.
        """
        self.controlador.set_escanear(False)
        self.controlador.borrar_todo()
        self.cambiar_tablero(self.controlador.obtener_tablero('TAB. INICIAL'))
        self.controlador.change_screen('inicio')
        Clock.unschedule(self.update)
        self.indice_prueba = 0
        self.controlador.stop_cronometro()
        self.pruebas_coordenadas = {}
        self.controlador.reiniciar_cronometro()

    def on_borrar_palabra(self, instance):
        """
        Al borrar palabra, borra tambien esta de la lista que se guarda para el resultado de la prueba
        """
        self.controlador.borrar_palabra()
        # Elimina la última palabra de las coordenadas
        if self.palabras_y_coordenadas:
            self.palabras_y_coordenadas.pop()

    def on_borrar_todo(self, instance):
        """
        Borra todo el texto y las palabras de la lista que se guarda para el resultado de la prueba
        """
        self.controlador.borrar_todo()
        self.palabras_y_coordenadas.clear()


    def on_reproducir(self, instance):
        """
        Al tocar sobre reproducir, se guardan los resulados de las pruebas si se ha completado correctamente.
        """
        self.controlador.reproducir_texto()
        if self.label.text.strip() == self.pruebas[self.indice_prueba]:
            self.indice_prueba += 1
            self.evaluate_test()
            self.controlador.stop_cronometro()
            self.controlador.set_escanear(False)
            Clock.unschedule(self.update)
        else:
            self.controlador.mensaje("Inténtalo de nuevo, escribe: " + self.pruebas[self.indice_prueba])
            self.on_borrar_todo(instance)

    
    def on_alarma(self, instance):
        """
        Cuando se toca sobre el botón de alarma, se reproduce la alarma.
        """
        self.controlador.reproducir_alarma()


    def cambiar_tablero(self, palabras):
        """
        Cambia el tablero actual por uno nuevo.
        """
        if palabras is None:
            palabras = []
        if self.tablero:
            self.layout_principal.remove_widget(self.tablero)
        self.tablero = Tablero(palabras, self.controlador, size_hint=(1, 0.8), pictos=self.controlador.get_pictogramas())
        self.layout_principal.add_widget(self.tablero, index=1)


    def update(self, dt):
        """
        Actualiza la pantalla de las pruebas.
        """
        # Obtener la frase actual
        self.label.text = self.controlador.get_frase()
        
        for dibujo in self.dibujos_mirada:
            self.canvas.remove(dibujo)

        self.dibujos_mirada = []

        if self.controlador.get_escanear() and self.controlador.get_screen() == 'tablerosprueb':

            # Emula el movimiento y clic
            if self.controlador.get_bloqueado():
                if not hasattr(self, 'pantalla_bloqueada'):
                    self.pantalla_bloqueada = PantallaBloqueada(self.controlador)
                    self.add_widget(self.pantalla_bloqueada)
            else:
                if hasattr(self, 'pantalla_bloqueada'):
                    self.remove_widget(self.pantalla_bloqueada)
                    del self.pantalla_bloqueada

                # Obtiene la posición de la mirada y el ear
                datos = self.controlador.obtener_posicion_mirada_ear()

                # Si no se detecta cara, no hacer nada
                if datos is None:
                    return
                
                # Desempaqueta los datos
                pos, click = datos
                pos = pos.flatten()
                x, y = pos

                # Emula el movimiento con las casillas 
                self.emular_movimiento_y_clic(x,y, click)

                # Normaliza las coordenadas
                x, y = pos * self.size
                
                # Dibuja el cursor
                tamaño_cruz = 20
                with self.canvas:
                    Color(1, 1, 1)
                    cruz1 = Line(points=[x - tamaño_cruz, y, x + tamaño_cruz, y], width=1)
                    cruz2 = Line(points=[x, y - tamaño_cruz, x, y + tamaño_cruz], width=1)
                    
                    # Normaliza contador_frames entre 0 y tamaño_cruz
                    radio_circulo = (self.contador_frames / self.frames_bloqueo) * tamaño_cruz

                    # Pinta un círculo con radio variable
                    circulo = Line(circle=(x, y, radio_circulo), width=2)

                # Añade los dibujos a la lista para eliminarlos en la próxima actualización
                self.dibujos_mirada.extend([cruz1, cruz2, circulo])

    def emular_movimiento_y_clic(self, x, y, click):       
        """
        Encargada de emular el movimiento y clic del usuario.
        """
        #Si la y es mayor que 0.2, casillas:
        if y > 0.17:
            casilla_ancho = 1 / self.tablero.cols
            casilla_alto = 0.8 / self.tablero.rows  # Quita el 0.2 inferior

            # Calcula a qué casilla corresponde la posición de la vista
            x = min(x, 1 - 1e-9)
            y = min(y, 1 - 1e-9)
            casilla_x = int(x / casilla_ancho)
            casilla_y = self.tablero.rows - 1 - int((y - 0.2) / casilla_alto)  # Resta 0.2 de y antes de calcular casilla_y

            # Convierte las coordenadas bidimensionales a un índice unidimensional
            indice_casilla = casilla_y * self.tablero.cols + casilla_x
        
        else: #Botones
            if x < 0.15:
                indice_casilla = self.tablero.cols * self.tablero.rows
            elif x > 0.55 and x < 0.70:
                indice_casilla = self.tablero.cols * self.tablero.rows + 1
            elif x >= 0.70 and x < 0.85:
                indice_casilla = self.tablero.cols * self.tablero.rows + 2
            elif x >= 0.85:
                indice_casilla = self.tablero.cols * self.tablero.rows + 4
            else: # Asi al clickar sobre el texto tambien reproduce el audio
                indice_casilla = self.tablero.cols * self.tablero.rows + 3

        
        # Guarda las posiciones del cursor
        self.cursor_positions.append((x, y))

        # Si la casilla es diferente a la casilla anterior, reinicia el contador de frames
        if self.casilla_anterior is None or indice_casilla != self.casilla_anterior:
            self.contador_frames = 0
        else:
            self.contador_frames += 1

        # Si el contador de frames llega a 30 (1 segundo a 30 FPS), bloquea la casilla
        if self.contador_frames >= self.frames_bloqueo:
            self.casilla_bloqueada = indice_casilla
            self.contador_frames = 0
            #se borran las posiciones del cursor menos las self.frames_bloqueo ultimas
            self.cursor_positions = self.cursor_positions[-self.frames_bloqueo:]


        # Actualiza el estado de todas las casillas
        for i, btn in enumerate(self.tablero.casillas + self.botones):
            if i == self.casilla_bloqueada:
                btn.state = 'down'
                if self.casilla_bloqueada == self.tablero.cols * self.tablero.rows + 3:
                    self.label.background_color = (0.3, 0.3, 0.3, 1)
            else:
                btn.state = 'normal'
                if i == self.tablero.cols * self.tablero.rows + 3:
                    self.label.background_color = (0.7, 0.7, 0.7, 1)


        # Si se hace click, se activa la casilla bloqueada
        if click and self.casilla_bloqueada is not None:
            if self.casilla_bloqueada < self.tablero.cols * self.tablero.rows:
                #Se apunta el tiempo que se tardo en elegir la primera palabra
                word = self.tablero.casillas[self.casilla_bloqueada].text
                if self.casilla_bloqueada == 0 and self.indice_prueba < 5:
                    pass
                else:
                    self.tablero.casillas[self.casilla_bloqueada].dispatch('on_press')
                self.palabras_y_coordenadas.append((word, self.casilla_bloqueada, self.cursor_positions.copy(), self.controlador.get_cronometro()))
                self.cursor_positions.clear()
            else:
                # Asegurar que el indice es correcto
                self.botones[min(self.casilla_bloqueada - self.tablero.cols * self.tablero.rows,4)].dispatch('on_press')
            self.casilla_bloqueada = None
        # Actualiza la casilla anterior
        self.casilla_anterior = indice_casilla 
        


    def evaluate_test(self):       
        """
        Guarda los resultados de la prueba en un archivo CSV.
        """
        with open(get_recurso('pruebas/pruebas.csv'), 'a', newline='') as f:
            writer = csv_writer(f)

            tiempo_total = self.controlador.get_cronometro()
            errores = self.controlador.get_errores()
            
            for palabra, indice, coordenadas, tiempo in self.palabras_y_coordenadas:
                writer.writerow([self.indice_prueba-1, palabra, indice, coordenadas, tiempo, tiempo_total, errores])
            
            writer.writerow(["-------", "-------", "-------", "-------", "-------", "-------", "-------"])

        # Limpia las palabras y coordenadas para la siguiente prueba
        self.palabras_y_coordenadas.clear()
    
        # Después de evaluar la prueba, pasar a la siguiente prueba o reiniciar
        self.pruebas_mensajes()

        