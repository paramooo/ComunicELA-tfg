from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from Custom import ButtonRnd
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Color, Line
from Tablero import Tablero, TableroPicto
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
import keyboard

class PantallaBloqueada(BoxLayout):
    def __init__(self, **kwargs):
        super(PantallaBloqueada, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)
        with self.canvas.before:
            Color(0, 0, 0, 0.85)  # color negro con alpha a 0.7
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        self.add_widget(Label(text='Mantenga los ojos cerrados 3 segundos para desbloquear', color=(1, 1, 1, 1)))

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class Tableros(Screen):
    def __init__(self, controlador, **kwargs):
        super(Tableros, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1)
        self.espacio_presionado = False


        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)

        # Layout principal
        self.layout_principal = BoxLayout(orientation='vertical')  # Cambia la orientación a vertical
        self.add_widget(self.layout_principal)
        
        # Tablero
        if self.controlador.get_pictogramas():
            self.tablero = TableroPicto(self.controlador.obtener_tablero('inicial'), self.controlador, size_hint=(1, 0.8))
        else:
            self.tablero = Tablero(self.controlador.obtener_tablero('inicial'), self.controlador, size_hint=(1, 0.8))

        self.layout_principal.add_widget(self.tablero)

                
        # Añade un espacio en blanco 
        espacio_blanco = BoxLayout(size_hint=(1, .05))

        # Layout de los botones
        layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, .15), spacing=10)
        self.layout_vertical = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        self.layout_vertical.add_widget(espacio_blanco)
        self.layout_vertical.add_widget(layout_botones)
        self.layout_principal.add_widget(self.layout_vertical)

        # El boton de inicio
        self.btn_inicio = ButtonRnd(text='Inicio', size_hint=(.12, 1), on_press= self.on_inicio, font_name='Texto')
        layout_botones.add_widget(self.btn_inicio)

        # Espacio para texto
        scroll = ScrollView(size_hint=(.4, 1), scroll_type=['bars', 'content'], bar_width=10)
        self.label = TextInput(
            text=self.controlador.get_frase(),
            # Limita el ancho del texto al ancho del widget
            size_hint_y=None,  # Esto permitirá que el TextInput se expanda a su tamaño natural
            height=Window.height * 0.2,  # Altura inicial del TextInput
            halign='left',  
            font_name='Texto', 
            font_size=40,
            background_color=(0, 0, 0, 0.4),
            foreground_color=(1, 1, 1, 1),
        )
        self.label.bind(on_text=self.on_text)  # Añade un evento para cuando el texto cambie
        scroll.add_widget(self.label)
        layout_botones.add_widget(scroll)


        # El boton para borrar una palabra
        self.btn_borrar_palabra = ButtonRnd(text='Borrar', size_hint=(.12, 1), on_press=self.on_borrar_palabra, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_palabra)

        # El boton para borrar todo el texto
        self.btn_borrar_todo = ButtonRnd(text='Borrar todo', size_hint=(.12, 1), on_press=self.on_borrar_todo, font_name='Texto')
        layout_botones.add_widget(self.btn_borrar_todo)

        # El boton para reproducir el texto
        self.btn_reproducir = ButtonRnd(text='Reproducir', size_hint=(.12, 1), on_press=self.on_reproducir, font_name='Texto')
        layout_botones.add_widget(self.btn_reproducir)

        # El boton de alarma
        self.btn_alarma = ButtonRnd(text='Alarma', size_hint=(.12, 1), on_press=self.on_alarma, font_name='Texto')
        layout_botones.add_widget(self.btn_alarma)

        #  ---------------------------------- Cronometro para las pruebas ----------------------------------------------
        self.cronometro = Label(size_hint=(.1, .05), pos_hint={'right':1, 'top':1})
        self.add_widget(self.cronometro)


        # Añade la tarea de actualización al reloj
        Clock.schedule_interval(self.update, 1.0 / 30.0)  

        # Variables para emular el movimiento y clic
        self.casilla_bloqueada = None
        self.contador_frames = 0
        self.casilla_anterior = None
        self.frames_bloqueo = 30
        self.botones = [self.btn_inicio, self.btn_borrar_palabra, self.btn_borrar_todo, self.btn_reproducir, self.btn_alarma]
        self.dibujos_mirada = []

    def on_text(self, instance, value):
        # TextInput siempre muestre la última línea de texto
        instance.scroll_y = 0

    # Cambia el tablero antes de entrar para evitar el salto de la vista
    def on_pre_enter(self, *args):
        self.cambiar_tablero(self.controlador.obtener_tablero('inicial'))

    # Funcion para escanear al entrar
    def on_enter(self, *args):
        self.controlador.set_escanear(True)
        self.controlador.set_bloqueado(False)


    # Parar de escanear al salir
    def on_inicio(self, instance):
        self.controlador.set_escanear(False)
        self.controlador.borrar_todo()
        self.controlador.reiniciar_cronometro()
        self.cambiar_tablero(self.controlador.obtener_tablero('inicial'))
        self.controlador.change_screen('inicio')



    # Funciones de los botones
    def on_borrar_palabra(self, instance):
        self.controlador.borrar_palabra()

    def on_borrar_todo(self, instance):
        self.controlador.borrar_todo()

    def on_reproducir(self, instance):
        #Aqui leer el texto en alto
        self.controlador.stop_cronometro(True)
        self.controlador.reproducir_texto()
    
    def on_alarma(self, instance):
        self.controlador.reproducir_alarma()


    # Cambia el tablero
    def cambiar_tablero(self, palabras):
        self.layout_principal.remove_widget(self.tablero)
        if self.controlador.get_pictogramas():
            self.tablero = TableroPicto(palabras, self.controlador, size_hint=(1, 0.8))
        else:
            self.tablero = Tablero(palabras, self.controlador, size_hint=(1, 0.8))
        self.layout_principal.add_widget(self.tablero, index=1)
        self.layout_principal.do_layout()


    # Actualiza la posición de la mirada
    def update(self, dt):
        # Obtener la frase actual
        self.label.text = self.controlador.get_frase()
        
        for dibujo in self.dibujos_mirada:
            self.canvas.remove(dibujo)

        self.dibujos_mirada = []

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
                    self.pantalla_bloqueada = PantallaBloqueada()
                    self.add_widget(self.pantalla_bloqueada)
            else:
                if hasattr(self, 'pantalla_bloqueada'):
                    self.remove_widget(self.pantalla_bloqueada)
                    del self.pantalla_bloqueada


                #---------------------------------------------------PARTE A COMENTAR PARA EL SOFTWARE FINAL-----------------------------------------
                # Si se pulsa el espacio, iniciar el cronómetro, si está activado, reiniciarlo
                espacio_actualmente_presionado = keyboard.is_pressed('space')
                if espacio_actualmente_presionado and not self.espacio_presionado:
                    if self.controlador.get_cronometro() == 0:
                        self.controlador.iniciar_cronometro()
                    else:
                        self.controlador.reiniciar_cronometro()
                self.espacio_presionado = espacio_actualmente_presionado

                # Actualiza el cronómetro
                tiempo = self.controlador.get_cronometro()
                self.cronometro.text = f'{int(tiempo):02}:{int((tiempo % 1) * 100):02}'
                #-------------------------------------------------------------------------------------------------------------------------------------



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
        #Si la y es mayor que 0.2, casillas:
        if y > 0.15:
            casilla_ancho = 1 / self.tablero.cols
            casilla_alto = 0.8 / self.tablero.rows  # Quita el 0.2 inferior

            # Calcula a qué casilla corresponde la posición de la vista
            casilla_x = int(x / casilla_ancho)
            casilla_y = self.tablero.rows - 1 - int((y - 0.2) / casilla_alto)  # Resta 0.2 de y antes de calcular casilla_y


            # Convierte las coordenadas bidimensionales a un índice unidimensional
            indice_casilla = casilla_y * self.tablero.cols + casilla_x
        
        else: #Botones
            if x < 0.12:
                indice_casilla = self.tablero.cols * self.tablero.rows
            elif x > 0.52 and x < 0.64:
                indice_casilla = self.tablero.cols * self.tablero.rows + 1
            elif x >= 0.64 and x < 0.76:
                indice_casilla = self.tablero.cols * self.tablero.rows + 2
            elif x >= 0.88:
                indice_casilla = self.tablero.cols * self.tablero.rows + 4
            else: # Asi al clickar sobre el texto tambien reproduce el audio
                indice_casilla = self.tablero.cols * self.tablero.rows + 3



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
            else:
                btn.state = 'normal'

        # Si se hace click, se activa la casilla bloqueada
        if click and self.casilla_bloqueada is not None:
            if self.casilla_bloqueada < self.tablero.cols * self.tablero.rows:
                self.tablero.casillas[self.casilla_bloqueada].dispatch('on_press')
            else:
                # Asegurar que el indice es correcto
                self.botones[min(self.casilla_bloqueada - self.tablero.cols * self.tablero.rows,4)].dispatch('on_press')
                print(self.casilla_bloqueada - self.tablero.cols * self.tablero.rows)
        # Actualiza la casilla anterior
        self.casilla_anterior = indice_casilla
