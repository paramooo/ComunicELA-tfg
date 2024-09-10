import random
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from KivyCustom.Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from KivyCustom.Custom import CustomSpinner
from KivyCustom.PopUp import CustPopup

class Recopilar(Screen):
    """
    Pantalla de de recopilación de datos de la aplicación.
    """
    def __init__(self, controlador, **kwargs):
        super(Recopilar, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.escaneado = False
        self.textos = ["Presione Recopilar para empezar",
                        "¡¡¡Gracias!!!, presiona Inicio para volver o Recopilar para volver a recopilar datos"]

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo()   )
        self.add_widget(self.fondo)
        
        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio, font_name='Texto')
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=self.textos[0], font_size=self.controlador.get_font_txts(), size_hint=(1, .8), font_name='Texto')
        self.layout.add_widget(self.texto_explicativo)



        # The image box
        self.image_box = Image(size_hint=(.2, .3), pos_hint={'right': 1, 'top': 0.1})
        self.layout.add_widget(self.image_box)

        # El boton de continuar
        self.btn_recopilar = ButtonRnd(text='Recopilar', size_hint=(.2, .1), pos_hint={'right': 1, 'top': 0}, on_press= self.on_recopilar, font_name='Texto')
        self.layout.add_widget(self.btn_recopilar)

        self.add_widget(self.layout)
        self.lanzado = False

    def on_enter(self, *args):
        """
        Al entrar en la pantalla, se reinician los datos de la pantalla y se añade la tarea de actualización al reloj.
        """
        self.escaneado = False
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
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)
        self.image_box.opacity = 1  
        self.btn_recopilar.disabled = False

    def on_inicio(self, *args):
        """
        Se ejecuta al presionar el boton de inicio, cambia a la pantalla de inicio y limpia los datos de la pantalla.
        """
        Clock.unschedule(self.update)
        Clock.unschedule(self.update_image_box)
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()
        self.descartar_datos()


    def on_recopilar(self, *args):
        """
        Al presionar en recopilar, se quita la imagen de la camara y se inicia la recopilación de datos.
        """
        self.image_box.opacity = 0  
        self.btn_recopilar.disabled = True

        self.controlador.on_recopilar()

    def update(self, dt):
        """
        Encargada de actualizar la pantalla de recopilación de datos segun el estado actual
        """
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
                if len(self.circulo_instr.children) > 1:
                    self.circulo_instr.children[1].pos = proxima_pos_r
        # Si no recopilando, pero ya recopilo datos, muestra el texto de agradecimiento
        elif self.escaneado:
            #Volver a mostrar la imagen
            self.image_box.opacity = 1
            self.btn_recopilar.disabled = False

            if not self.lanzado:
                self.lanzado = True
                #popup con opcion para guardar o descartar
                CustPopup("¡Gracias por su colaboración!", self.guardar_datos, (0.5,0.5), self.controlador, func_saltar=self.descartar_datos).open()

        else:
            # Si no recolecto datos aun, muestra el texto explicativo normal
            self.texto_explicativo.text = self.textos[0]

    def guardar_datos(self, *args):
        """
        Guarda los datos finalmente.
        """
        self.controlador.guardar_final()
        self.escaneado = False
        self.lanzado = False
    
    def descartar_datos(self, *args):
        """
        Descarta los datos recopilados.
        """
        self.controlador.descartar_datos()
        self.escaneado = False
        self.lanzado = False



    def update_image_box(self, dt):
        """
        Actualiza la imagen de la cámara 
        """
        frame = self.controlador.get_frame_editado()
        if frame is None:
            return
        
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.image_box.texture = texture



