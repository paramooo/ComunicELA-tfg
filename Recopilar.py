import random
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock

class Recopilar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Recopilar, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 

        self.escaneado = False
        self.textos = ["Presiona Recopilar para empezar a recopilar datos, debes mirar fijamente a la pelota roja.\n" + 
                        "Cuando presiones Recopilar, en 5 segundos empezara a moverse por toda la pantalla.\n" + 
                        "Mirala fijamente hasta que termine de moverse, ¡Muchas gracias! (no tardará más de 3 minutos)",
                        "¡¡¡Muchas gracias!!!, presiona Inicio para volver o Recopilar para volver a recopilar datos"]

        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio)
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=self.textos[0], font_size=self.controlador.get_font_txts(), size_hint=(1, .8))
        self.layout.add_widget(self.texto_explicativo)

        # El boton de continuar
        btn_recopilar = ButtonRnd(text='Recopilar', size_hint=(.2, .1), pos_hint={'right': 1, 'top': 0}, on_press= self.on_recopilar)
        self.layout.add_widget(btn_recopilar)



        self.add_widget(self.layout)

    def on_enter(self, *args):
        self.controlador.reiniciar_datos_r()
        # Añade la tarea de actualización al reloj
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS  
        # Se crea el circulo rojo y se añade
        self.circulo = Ellipse(pos=(0,0), size=(50, 50))
        with self.canvas:
            Color(1, 0, 0)  # Rojo
            self.circulo_instr = InstructionGroup()
            self.circulo_instr.add(self.circulo)
            if self.circulo_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.circulo_instr)

    def on_inicio(self, *args):
        # Detiene el evento del reloj programado para update
        Clock.unschedule(self.update)

        # Cambia a la pantalla de inicio
        self.controlador.change_screen('inicio')

        # Limpia las instrucciones de gráficos del círculo
        self.circulo_instr.clear()


    # Funcion para el boton recopilar, pone recopilar a true e inicia la cuanta atras
    def on_recopilar(self, *args):
        self.controlador.recopilar_datos()
        self.controlador.on_recopilar()

    def update(self, dt):
        #Si recopilar, actualiza el texto explicativo a la cuenta atras
        if self.controlador.get_recopilando():
            # Actualiza el texto explicativo con el contador del controlador
            contador = self.controlador.get_contador_reco()
            if contador != 0:
                self.texto_explicativo.text = str(contador)
            else:
                self.texto_explicativo.text = ""

            # Si el contador del controlador es 0, empieza a recopilar datos
            if contador == 0:
                self.escaneado = True
                # Obtiene el tamaño de la pantalla
                tamano_pantalla = self.get_root_window().size

                # Obtiene la próxima posición del círculo del controlador
                proxima_pos_r = self.controlador.actualizar_pos_circle_r(tamano_pantalla)

                # Actualiza la posición del círculo en la vista
                self.circulo_instr.children[1].pos = proxima_pos_r

        # Si no recopilar, pero ya recopilo datos, muestra el texto de agradecimiento
        elif self.escaneado:
            self.texto_explicativo.text = self.textos[1]
        else:
            # Si no recolecto datos aun, muestra el texto explicativo normal
            self.texto_explicativo.text = self.textos[0]
