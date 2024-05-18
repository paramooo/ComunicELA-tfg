import random
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from Custom import CustomSpinner

class Recopilar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Recopilar, self).__init__(**kwargs)
        self.fichero = "0"
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.porcentajeDisp = 0.15

        self.escaneado = False
        self.textos = ["Primero elige el tipo de recopilado:\n" + 
                        "-  0 -> muy bien iluminado, mantener la cabeza casi quieta y distancia 50/60cm - (15%)\n" +
                        "-  1 -> sin importar demasiado la luz, mantener la cabeza por el centro y distancia maxima 70cm - (30%)\n" +
                        "ASEGURATE DE QUE LA CAMARA ESTA A LA ALTURA DE LOS OJOS\n" + 
                        "Coloca tu cabeza a la distancia requerida de la camara y cuadra tu cabeza en el cuadro si te moviste despues del calibrado\n" +  
                        "A continuación debes mirar fijamente a la pelota roja.\n" + 
                        "Cuando presiones Recopilar, en 5 segundos esta empezara a moverse por toda la pantalla.\n" + 
                        "Mirala fijamente hasta que termine de moverse, ¡Muchas gracias! (no tardará más de 3 minutos)",
                        "¡¡¡Gracias!!!, presiona Inicio para volver o Recopilar para volver a recopilar datos"]

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



        # The image box
        self.image_box = Image(size_hint=(.2, .3), pos_hint={'right': 1, 'top': 0.1})
        self.layout.add_widget(self.image_box)

        # El boton de continuar
        self.btn_recopilar = ButtonRnd(text='Recopilar', size_hint=(.2, .1), pos_hint={'right': 1, 'top': 0}, on_press= self.on_recopilar, font_name='Texto')
        self.layout.add_widget(self.btn_recopilar)

        # Menu de seleccion de fichero
        self.camera_spinner = CustomSpinner(
        text='Seleccione fichero',
        values=["0","1"],
        size_hint=(0.2, 0.1),
        pos_hint={'right': 1, 'top': 0.1},
        )
        self.layout.add_widget(self.camera_spinner)

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
        # Schedule the update of the image box every 1/30 seconds
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)
        self.image_box.opacity = 1  # Show the image box
        self.btn_recopilar.disabled = False

    def on_inicio(self, *args):
        # Cambia a la pantalla de inicio
        self.controlador.change_screen('inicio')
        # Limpia las instrucciones de gráficos del círculo
        self.circulo_instr.clear()


    # Funcion para el boton recopilar, pone recopilar a true e inicia la cuanta atras
    def on_recopilar(self, *args):
        # Obtiene el fichero seleccionado
        self.fichero = self.camera_spinner.text
        if self.fichero == 'Seleccione fichero':
            self.controlador.mensaje('Seleccione un fichero')
            return
        self.image_box.opacity = 0  # Hide the image box
        self.btn_recopilar.disabled = True

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
                proxima_pos_r = self.controlador.actualizar_pos_circle_r(tamano_pantalla, self.fichero)

                # Actualiza la posición del círculo en la vista
                if len(self.circulo_instr.children) > 1:
                    self.circulo_instr.children[1].pos = proxima_pos_r
        # Si no recopilar, pero ya recopilo datos, muestra el texto de agradecimiento
        elif self.escaneado:
            self.texto_explicativo.text = self.textos[1]
            #Volver a mostrar la imagen
            self.image_box.opacity = 1
            self.btn_recopilar.disabled = False

        else:
            # Si no recolecto datos aun, muestra el texto explicativo normal
            self.texto_explicativo.text = self.textos[0]


    def update_image_box(self, dt):
            # Poner la zona donde se puede mover la persona
            self.porcentajeDisp = 0.15 if self.camera_spinner.text == "0" else 0.30

            # Only update the image box in calibration state 0
            frame = self.controlador.get_frame_editado(self.porcentajeDisp)
            if frame is None:
                return
            
            # Convert the frame to a texture
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')

            # Invertir la imagen verticalmente
            texture.flip_vertical()
            self.image_box.texture = texture


    def on_leave(self, *args):
        Clock.unschedule(self.update)
        Clock.unschedule(self.update_image_box)
