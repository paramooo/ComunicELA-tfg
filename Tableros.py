from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from Custom import ButtonRnd
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics import Color, Line
from Tablero import Tablero

class Tableros(Screen):
    def __init__(self, controlador, **kwargs):
        super(Tableros, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 

        # Layout principal
        self.layout_principal = BoxLayout(orientation='vertical')  # Cambia la orientación a vertical
        self.add_widget(self.layout_principal)
        
        # Tablero
        self.tablero = Tablero(self.controlador.obtener_tablero('inicial'), self.controlador)
        self.layout_principal.add_widget(self.tablero)

                
        # Añade un espacio en blanco 
        espacio_blanco = BoxLayout(size_hint=(1, .03))  # Elimina pos_hint
        # with espacio_blanco.canvas:
        #     Color(1, 1, 1)
        #     Line(points=[espacio_blanco.x, espacio_blanco.y + espacio_blanco.height / 2, espacio_blanco.right, espacio_blanco.y + espacio_blanco.height / 2], width=2)
        self.layout_principal.add_widget(espacio_blanco)


        # Layout de los botones
        layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, .2), spacing=10)
        self.layout_principal.add_widget(layout_botones)

        # El boton de inicio
        self.btn_inicio = ButtonRnd(text='Inicio', size_hint=(.15, 1), on_press= self.on_inicio)
        layout_botones.add_widget(self.btn_inicio)

        # Espacio para texto
        self.label = Label(text=self.controlador.get_frase(), size_hint=(.4, 1))
        layout_botones.add_widget(self.label)

        # El boton para borrar una palabra
        self.btn_borrar_palabra = ButtonRnd(text='<', size_hint=(.15, 1), on_press=self.on_borrar_palabra)
        layout_botones.add_widget(self.btn_borrar_palabra)

        # El boton para borrar todo el texto
        self.btn_borrar_todo = ButtonRnd(text='<|', size_hint=(.15, 1), on_press=self.on_borrar_todo)
        layout_botones.add_widget(self.btn_borrar_todo)

        # El boton para reproducir el texto
        self.btn_reproducir = ButtonRnd(text='Reproducir', size_hint=(.15, 1), on_press=self.on_reproducir)
        layout_botones.add_widget(self.btn_reproducir)

        # Añade la tarea de actualización al reloj
        Clock.schedule_interval(self.update, 1.0 / 30.0)  

        # Variables para emular el movimiento y clic
        self.casilla_bloqueada = None
        self.contador_frames = 0
        self.casilla_anterior = None
        self.frames_bloqueo = 10
        self.botones = [self.btn_inicio, self.btn_borrar_palabra, self.btn_borrar_todo, self.btn_reproducir]



    # Funcion para escanear al entrar
    def on_enter(self, *args):
        self.controlador.set_escanear(True)

    # Parar de escanear al salir
    def on_inicio(self, instance):
        self.controlador.set_escanear(False)
        self.controlador.borrar_todo()
        self.cambiar_tablero(self.controlador.obtener_tablero('inicial'))
        self.controlador.change_screen('inicio')



    # Funciones de los botones
    def on_borrar_palabra(self, instance):
        self.controlador.borrar_palabra()

    def on_borrar_todo(self, instance):
        self.controlador.borrar_todo()

    def on_reproducir(self, instance):
        #Aqui leer el texto en alto
        self.controlador.reproducir_texto()


    # Cambia el tablero
    def cambiar_tablero(self, palabras):
        self.layout_principal.remove_widget(self.tablero)
        self.tablero = Tablero(palabras, self.controlador)
        self.layout_principal.add_widget(self.tablero, index=1)
        


    # Actualiza la posición de la mirada
    def update(self, dt):
        if self.controlador.get_escanear() and self.controlador.get_screen() == 'tableros':
            # Obtiene la posición de la mirada y el ear
            datos = self.controlador.obtener_posicion_mirada_ear()

            # Si no se detecta cara, no hacer nada
            if datos is None:
                self.controlador.mensaje("No se detecta cara")
                return
            
            # Desempaqueta los datos
            pos, click = datos

            pos = pos.flatten()

            # Actualiza la posición de la mirada
            x, y = pos*self.size
            
            # Si parpadea, se cogera el de 30 frames atras
            self.emular_movimiento_y_clic(x,y, click)



    def emular_movimiento_y_clic(self, x, y, click):
        #Si la y es mayor que 0.2, casillas:
        if y > self.size[1] * 0.2:
            # Calcula el tamaño de cada casilla
            casilla_ancho = self.size[0] / self.tablero.cols
            casilla_alto = (self.size[1] * 0.8) / self.tablero.rows  # Quita el 0.2 inferior

            # Calcula a qué casilla corresponde la posición de la vista
            casilla_x = int(x / casilla_ancho)
            casilla_y = self.tablero.rows - 1 - int(y / casilla_alto)  # Invierte el cálculo del índice y

            # Convierte las coordenadas bidimensionales a un índice unidimensional
            indice_casilla = casilla_y * self.tablero.cols + casilla_x
        
        else: #Botones
            if x <  self.size[0] * 0.3:
                indice_casilla = self.tablero.cols * self.tablero.rows
            elif x > self.size[0] * 0.55 and x < self.size[0] * 0.7:
                indice_casilla = self.tablero.cols * self.tablero.rows + 1
            elif x > self.size[0] * 0.7 and x < self.size[0] * 0.85:
                indice_casilla = self.tablero.cols * self.tablero.rows + 2
            else:
                indice_casilla = self.tablero.cols * self.tablero.rows + 3


        # Si la casilla es diferente a la casilla anterior, reinicia el contador de frames
        if self.casilla_anterior is None or indice_casilla != self.casilla_anterior:
            self.contador_frames = 0
        else:
            self.contador_frames += 1

        # Si el contador de frames llega a 30 (1 segundo a 30 FPS), bloquea la casilla
        if self.contador_frames >= self.frames_bloqueo:
            self.casilla_bloqueada = indice_casilla

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
                self.botones[self.casilla_bloqueada - self.tablero.cols * self.tablero.rows].dispatch('on_press')

        # Actualiza la casilla anterior
        self.casilla_anterior = indice_casilla
