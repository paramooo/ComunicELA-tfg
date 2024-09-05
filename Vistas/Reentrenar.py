from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock
from kivy.uix.widget import Widget
from KivyCustom.PopUp import CustPopup

class Reentrenar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Reentrenar, self).__init__(**kwargs)
        self.controlador = controlador
        self.escaneado = False

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo()   )
        self.add_widget(self.fondo)

        self.layout = BoxLayout(orientation='vertical')


        # El texto explicativo
        self.texto_explicativo = Label(text="", font_size=30, size_hint=(1, .8), font_name='Texto', color=(1, 1, 1, 1))
        self.layout.add_widget(self.texto_explicativo)
        self.layout.add_widget(Widget(size_hint_y=0.1))

        self.add_widget(self.layout)
        self.lanzado = False

    def on_pre_enter(self, *args):
        self.texto_explicativo.text = ""


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
        self.controlador.set_reentrenando(True)
        self.controlador.on_recopilar()
        self.lanzado = False



    def on_inicio(self, *args):
        # Cambia a la pantalla de inicio
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()
        self.controlador.set_reentrenando(False)
        Clock.unschedule(self.update)
        self.lanzado = False
        

    def on_leave(self, *args):
        # Elimina la tarea de actualización del reloj
        Clock.unschedule(self.update)



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
            # Si el porcentaje de reentrenamiento es -1, muestra un mensaje de error
            if self.controlador.get_reent_porcentaje() == -1:
                if not self.lanzado:
                    self.lanzado = True
                    CustPopup(self.controlador.get_string('error_recop_ree'), self.on_inicio, (0.5,0.5), controlador=self.controlador).open()

            # Si el porcentaje de reentrenamiento es 100, muestra el avance
            elif self.controlador.get_reent_porcentaje() < 100:
                self.texto_explicativo.text = self.controlador.get_string('reentrenando') + f"... {self.controlador.get_reent_porcentaje()}%"
            
            # Si el porcentaje de reentrenamiento es 100, muestra un mensaje de finalización
            else:
                if self.controlador.get_optimizando():
                    self.texto_explicativo.text = self.controlador.get_string('optimizando') + f"... {self.controlador.get_progreso_opt()}%"
                else:
                    if not self.lanzado:
                        self.lanzado = True
                        self.controlador.sumar_reentrenamiento()
                        CustPopup(self.controlador.get_string('reentrenamiento_completado'), self.on_inicio, (0.5,0.5), controlador=self.controlador).open()
