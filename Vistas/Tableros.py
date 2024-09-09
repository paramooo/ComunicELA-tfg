from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from KivyCustom.Custom import ButtonRnd, CustomTextInput
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Color, Line
from KivyCustom.Tablero import Tablero
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.graphics import Rectangle

class PantallaBloqueada(BoxLayout):
    """
    Fondo semitransparente que se superpone a la pantalla para bloquear la interacción con los elementos.
    """
    def __init__(self, controlador, **kwargs):
        super(PantallaBloqueada, self).__init__(**kwargs)
        self.controlador = controlador
        self.orientation = 'vertical'
        self.size_hint = (1, 1)
        with self.canvas.before:
            Color(0, 0, 0, 0.85)  # color negro con alpha a 0.7
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        self.add_widget(Label(text=self.controlador.get_string("mensaje_descanso"), color=(1, 1, 1, 1)))

    def _update_rect(self, instance, value):
        """
        Actualiza el tamaño al de la pantalla
        """
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class Tableros(Screen):
    """
    Tableros interactivos de la aplicación.
    """
    def __init__(self, controlador, **kwargs):
        super(Tableros, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1)


        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo()   )
        self.add_widget(self.fondo)

        # Layout principal
        self.layout_principal = BoxLayout(orientation='vertical')  # Cambia la orientación a vertical
        self.add_widget(self.layout_principal)
                
        # Añade un espacio en blanco 
        espacio_blanco = BoxLayout(size_hint=(1, .03))

        # Layout de los botones
        layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, .17), spacing=10)
        self.layout_vertical = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        self.layout_vertical.add_widget(espacio_blanco)
        self.layout_vertical.add_widget(layout_botones)
        self.layout_principal.add_widget(self.layout_vertical)

        # El boton de inicio
        self.btn_inicio = ButtonRnd(text=self.controlador.get_string("inicio"), size_hint=(.15, 1), on_press= self.on_inicio, font_name='Texto')
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

        self.label.bind(on_text=self.on_text)  # Añade un evento para cuando el texto cambie
        self.label.bind(on_touch_down = self.on_label_touch_down)
        scroll.add_widget(self.label)
        layout_botones.add_widget(scroll)


        # El boton para borrar una palabra
        self.btn_borrar_palabra = ButtonRnd(text=self.controlador.get_string("borrar"), size_hint=(.15, 1), on_press=self.on_borrar_palabra, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_palabra)

        # El boton para borrar todo el texto
        self.btn_borrar_todo = ButtonRnd(text=self.controlador.get_string("borrar_todo"), size_hint=(.15, 1), on_press=self.on_borrar_todo, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_todo)

        # # El boton para reproducir el texto
        self.btn_reproducir = ButtonRnd(text=self.controlador.get_string("reproducir"), size_hint=(.12, 1), on_press=self.on_reproducir, font_name='Texto')
        # layout_botones.add_widget(self.btn_reproducir)        

        # El boton de alarma
        self.btn_alarma = ButtonRnd(text=self.controlador.get_string("alarma"), size_hint=(.15, 1), on_press=self.on_alarma, font_name='Texto')
        layout_botones.add_widget(self.btn_alarma)

        # Variables para emular el movimiento y clic
        self.casilla_bloqueada = None
        self.contador_frames = 0
        self.casilla_anterior = None
        self.frames_bloqueo = 30
        self.botones = [self.btn_inicio, self.btn_borrar_palabra, self.btn_borrar_todo, self.btn_reproducir, self.btn_alarma]
        self.dibujos_mirada = []
    
    def on_text(self, instance, value):
        """
        Hace que el texto se desplace hacia arriba cuando se añade una nueva línea
        """
        instance.scroll_y = 0

    def on_pre_enter(self, *args):
        """
        Establece el tablero inicial antes de entrar en la pantalla.
        """
        tab_incial = self.controlador.obtener_tablero_inicial()
        self.cambiar_tablero(self.controlador.obtener_tablero(tab_incial))
        self.label.text = self.controlador.get_frase()

    def on_enter(self, *args):
        """
        Cuando entra, se añade la tarea de actualización al reloj, se activa el escaneo y desactiva el bloqueo.
        """
        Clock.schedule_interval(self.update, 1.0 / 30.0)  
        self.controlador.set_escanear(True)
        self.controlador.set_bloqueado(False)

    def on_inicio(self, instance):
        """
        Cambia a la pantalla de inicio.
        """
        self.controlador.set_escanear(False)
        self.controlador.borrar_todo()
        tab_incial = self.controlador.obtener_tablero_inicial()
        self.cambiar_tablero(self.controlador.obtener_tablero(tab_incial))
        self.controlador.change_screen('inicio')
        Clock.unschedule(self.update)

    def on_label_touch_down(self, instance, touch):
        """
        Al tocar sobre el boton de reproducir, se activa la reproducción del texto.
        """
        self.on_reproducir(instance)

    def on_borrar_palabra(self, instance):
        """
        Notifica al controlador que se ha pulsado el botón de borrar palabra.
        """
        self.controlador.borrar_palabra()

    def on_borrar_todo(self, instance):
        """
        Notifica al controlador que se ha pulsado el botón de borrar todo.
        """
        self.controlador.borrar_todo()

    def on_reproducir(self, instance):
        """
        Notifica al controlador que se ha pulsado el botón de reproducir.
        """
        self.controlador.reproducir_texto()
    
    def on_alarma(self, instance):
        """
        Notifica al controlador que se ha pulsado el botón de alarma.
        """
        self.controlador.reproducir_alarma()


    def cambiar_tablero(self, palabras):
        """
        Cambia el tablero actual por el que se le pasa por parámetro.

        Args:
            palabras (list): Lista de palabras a mostrar en el tablero
        """
        if hasattr(self, 'tablero'):
            self.layout_principal.remove_widget(self.tablero)
        self.tablero = Tablero(palabras, self.controlador, size_hint=(1, 0.8), pictos=self.controlador.get_pictogramas())
        self.layout_principal.add_widget(self.tablero, index=1)
        self.layout_principal.do_layout()


    def update(self, dt):
        """
        Se encarga de actualizar la pantalla, emulando el movimiento y clic de la mirada.
        """
        # Obtener la frase actual
        self.label.text = self.controlador.get_frase()
        
        for dibujo in self.dibujos_mirada:
            self.canvas.remove(dibujo)

        self.dibujos_mirada = []
        if self.controlador.get_camara_seleccionada() is not None:
            if self.controlador.get_escanear() and self.controlador.get_screen() == 'tableros':

                # Obtiene la posición de la mirada y el ear
                datos = self.controlador.obtener_posicion_mirada_ear()

                # Si no se detecta cara, no hacer nada
                if datos is None:
                    return
                
                # Desempaqueta los datos
                pos, click = datos
                pos = pos.flatten()
                x, y = pos

                # Emula el movimiento y clic
                if self.controlador.get_bloqueado():
                    if not hasattr(self, 'pantalla_bloqueada'):
                        self.pantalla_bloqueada = PantallaBloqueada(controlador=self.controlador)
                        self.add_widget(self.pantalla_bloqueada)
                else:
                    if hasattr(self, 'pantalla_bloqueada'):
                        self.remove_widget(self.pantalla_bloqueada)
                        del self.pantalla_bloqueada

                    # Emula el movimiento con las casillas 
                    self.emular_movimiento_y_clic(x,y, click)

                    # Normaliza las coordenadas
                    x, y = pos*self.size

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
        Emula el movimiento y clic de la mirada en el tablero.
        """      
        if y > 0.17:
            casilla_ancho = 1 / self.tablero.cols
            casilla_alto = 0.8 / self.tablero.rows  # Quita el 0.2 inferior

            # Limita las coordenadas 
            x = min(x, 1 - 1e-9)
            y = min(y, 1 - 1e-9)

            # Calcula a qué casilla corresponde la posición de la vista
            casilla_x = int(x / casilla_ancho)
            casilla_y = self.tablero.rows - 1 - int((y - 0.2) / casilla_alto)  # Resta 0.2 de y antes de calcular casilla_y


            # Convierte las coordenadas bidimensionales a un índice unidimensional
            indice_casilla = casilla_y * self.tablero.cols + casilla_x
        
        else: #Botones
            if x < 0.15:
                indice_casilla = self.tablero.cols * self.tablero.rows #Boton inicio
            elif x > 0.55 and x < 0.70:
                indice_casilla = self.tablero.cols * self.tablero.rows + 1 #Boton borrar palabra
            elif x >= 0.70 and x < 0.85:
                indice_casilla = self.tablero.cols * self.tablero.rows + 2 #Boton borrar todo
            elif x >= 0.85:
                indice_casilla = self.tablero.cols * self.tablero.rows + 4 #Boton alarma
            else:
                indice_casilla = self.tablero.cols * self.tablero.rows + 3 #Barra de texto



        # Si la casilla es diferente a la casilla anterior, reinicia el contador de frames
        if self.casilla_anterior is None or indice_casilla != self.casilla_anterior:
            self.contador_frames = 0
        else:
            self.contador_frames += 1

        # Si el contador de frames llega a 30 (1 segundo a 30 FPS), bloquea la casilla
        if self.contador_frames >= self.frames_bloqueo:
            self.casilla_bloqueada = indice_casilla
            self.contador_frames = 0

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
                self.tablero.casillas[self.casilla_bloqueada].dispatch('on_press')
            else:
                # Asegurar que el indice es correcto
                self.botones[min(self.casilla_bloqueada - self.tablero.cols * self.tablero.rows,4)].dispatch('on_press')
            self.casilla_bloqueada = None
        # Actualiza la casilla anterior
        self.casilla_anterior = indice_casilla
