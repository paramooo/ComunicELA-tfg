from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import InstructionGroup
import numpy as np

class Test(Screen):
    def __init__(self, controlador, **kwargs):
        super(Test, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.controlador.set_escanear(False)

        texto_central = ("Mire al rededor de la pantalla y parpadee para confirmar el calibrado\n"+
                        "Es un modelo en desarrollo, cuantos mas datos recopilemos, mejor funcionara en un futuro\n"+
                        "El punto rojo deberia seguir su mirada, al parpadear debería escuchar un sonido de click y ver el punto verde")

        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio)
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=texto_central, font_size=self.controlador.get_font_txts(),halign='center', size_hint=(1, .8))
        self.layout.add_widget(self.texto_explicativo)

        # Añade la tarea de actualización al reloj
        Clock.schedule_interval(self.update, 1.0 / 60.0)  # 60 FPS

        self.add_widget(self.layout)



    # Funcion para dibujar el circulo amarillo una vez abierta la ventana(para centrarlo bien)
    def on_enter(self, *args):  
        # Añade la tarea de actualización al reloj
        #Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS  
        
        # Se crea el circulo rojo y se añade
        self.circulo = Ellipse(pos=self.center, size=(50, 50))
        with self.canvas:
            self.circulo_color = Color(1, 0, 0)  # Rojo
            self.circulo_instr = InstructionGroup()
            self.circulo_instr.add(self.circulo_color)
            self.circulo_instr.add(self.circulo)
            if self.circulo_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.circulo_instr)
        self.controlador.set_escanear(True)

    # Funcion para dejar de escanear y volver al inicio
    def on_inicio(self, *args):
        self.controlador.set_escanear(False)
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()

    # Funcion para actualizar la posición del círculo y el color
    def update(self, dt):
        # Si se ha activado el escaneo y estamso en esta pantalla
        if self.controlador.get_escanear() and self.controlador.get_screen() == 'test':
            # Obtiene la posición de la mirada y el ear
            datos = self.controlador.obtener_posicion_mirada_ear()

            # Si no se detecta cara, no hacer nada
            if datos is None:
                self.controlador.mensaje("No se detecta cara")
                return
            
            # Desempaqueta los datos
            proxima_pos_t, click = datos

            # Actualiza el color del círculo
            if click == 0:  # Rojo
                self.circulo_color.rgba = (1, 0, 0, 1)
            elif click == 1:  # Verde
                self.circulo_color.rgba = (0, 1, 0, 1)

            # La posición del circulo no excede las dimensiones de la pantalla
            proxima_pos_t = np.minimum(proxima_pos_t*self.size, self.size - np.array([50, 50]))
            
            #Indice 2 ya que tiene el color en 1, no como recopilar:
            self.circulo_instr.children[2].pos =  (proxima_pos_t).flatten()


                