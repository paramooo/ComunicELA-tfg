from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import InstructionGroup

class Test(Screen):
    def __init__(self, controlador, **kwargs):
        super(Test, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.escanear = False

        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio)
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text="\"Mire al rededor de la pantalla\" y parpadee para confirmar el calibrado", font_size=self.controlador.get_font_txts(), size_hint=(1, .8))
        self.layout.add_widget(self.texto_explicativo)

        # Añade la tarea de actualización al reloj
        Clock.schedule_interval(self.update, 1.0 / 60.0)  # 60 FPS

        self.add_widget(self.layout)



    # Funcion para dibujar el circulo amarillo una vez abierta la ventana(para centrarlo bien)
    def on_enter(self, *args):  
        # Se crea el circulo rojo y se añade
        with self.canvas:
            self.circulo = Ellipse(pos=self.center, size=(100, 100))

            self.circulo_color = Color(1, 0, 0)  # Rojo
            self.circulo_instr = InstructionGroup()
            self.circulo_instr.add(self.circulo_color)
            self.circulo_instr.add(self.circulo)
            if self.circulo_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.circulo_instr)
        self.escanear = True

    # Funcion para dejar de escanear y volver al inicio
    def on_inicio(self, *args):
        self.escanear = False
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()

    
    # Funcion para actualizar la posición del círculo y el color
    def update(self, dt):
        if self.escanear:
            # Obtiene las nuevas coordenadas del controlador
            #x, y = self.controlador.get_coordinates()
            # Actualiza la posición del círculo
            #self.circulo.pos = (x * self.width, y * self.height)

            # Obtiene el color del controlador
            color = self.controlador.get_parpadeo()

            # Actualiza el color del círculo
            if color == 0:  # Rojo
                self.circulo_color.rgba = (1, 0, 0, 1)
            elif color == 1:  # Verde
                self.circulo_color.rgba = (0, 1, 0, 1)

            # distancias = self.controlador.get_distancias_ojos()
            # if distancias is not None:
            #     self.texto_explicativo.text = "Distancia ojo izquierdo: " + str(distancias[0][0]) + "\nDistancia ojo derecho: " + str(distancias[1][0])
