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
        self.label = Label(text=self.controlador.get_frase(), size_hint=(.6, 1))
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
        Clock.schedule_interval(self.update, 1.0 / 60.0)  # 60 FPS


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
        buttons = [self.btn_inicio, self.btn_borrar_palabra, self.btn_borrar_todo, self.btn_reproducir] + self.tablero.casillas
        for btn in buttons:
            if btn.collide_point(x, y):
                btn.state = 'down'
                if click:
                    btn.dispatch('on_press')
            else:
                btn.state = 'normal'
