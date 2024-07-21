from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock


class Reentrenar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Reentrenar, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.escaneado = False
        

        self.textos = ["Al pulsar Reentrenar, el usuario debe mirar a la pelota roja hasta que termine de moverse.", 
                       "Reentrenamiento completado, presiona Inicio para volver, o Reentrenar para volver a realizar el proceso"]

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)

        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio, font_name='Texto')
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=self.textos[0], font_size=self.controlador.get_font_txts(), size_hint=(1, .8), font_name='Texto')
        self.layout.add_widget(self.texto_explicativo)
        
        # El boton de continuar
        self.btn_recopilar = ButtonRnd(text='Reentrenar', size_hint=(.2, .1), pos_hint={'right': 1, 'top': 0}, on_press= self.on_recopilar, font_name='Texto')
        self.layout.add_widget(self.btn_recopilar)

        self.add_widget(self.layout)

    def on_enter(self, *args):
        self.escaneado = False
        self.controlador.reiniciar_datos_r()
        self.controlador.reiniciar_datos_ree()
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
        # Schedule the update of the image box every 1/30 seconds
        self.btn_recopilar.disabled = False

    def on_inicio(self, *args):
        # Cambia a la pantalla de inicio
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()
        self.controlador.set_reentrenando(False)

    def on_leave(self, *args):
        # Elimina la tarea de actualización del reloj
        Clock.unschedule(self.update)


    def on_recopilar(self, *args):
        self.btn_recopilar.disabled = True
        self.controlador.set_reentrenando(True)
        self.controlador.on_recopilar()


    def update(self, dt):
        # Si se está recopilando datos
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
                if len(self.circulo_instr.children) > 1:
                    self.circulo_instr.children[1].pos = proxima_pos_r
        # Si no recopilar, pero ya recopilo datos, muestra el texto de agradecimiento
        elif self.escaneado:
            self.texto_explicativo.text = self.textos[1]
            #Volver a mostrar la imagen
            self.btn_recopilar.disabled = False

        else:
            # Si no recolecto datos aun, muestra el texto explicativo normal
            self.texto_explicativo.text = self.textos[0]