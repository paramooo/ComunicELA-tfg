from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, Ellipse, Line
from kivy.graphics import InstructionGroup
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2

class Calibrar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Calibrar, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.porcentajeDisp = 0.10  

        # Textos de calibración
        self.textos_calibracion = [
            'Cuadrar la cruz con el punto medio entre las cejas y la cabeza recta, posteriormente mire fijamente el punto \n' + 
            'amarillo y mientras tanto haga click en el boton Continuar (mirando al punto)',
            'Ahora porfavor cierre los ojos y mientras tanto vuelva a presionar el boton de continuar (con los ojos cerrados)',
            'Calibración completada, presione continuar para ir al inicio'
        ]     

        # Creamos la caja para meter todo
        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio)
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=self.textos_calibracion[0], font_size=self.controlador.get_font_txts(), size_hint=(1, .1), pos_hint={'top': .7})
        self.layout.add_widget(self.texto_explicativo)
        
        # The image box
        self.image_box = Image(size_hint=(.5, .4), pos_hint={'center_x': .5, 'top': 1})
        self.layout.add_widget(self.image_box)

        # El boton de comenzar
        btn_comenzar = ButtonRnd(text='Continuar', size_hint=(.2, .1), pos_hint={'right': 1, 'top': 0}, on_press=self.on_continuar)
        self.layout.add_widget(btn_comenzar)

        # Add the layout to the screen
        self.add_widget(self.layout)


    # Funcion para dibujar el circulo amarillo una vez abierta la ventana(para centrarlo bien)
    def on_enter(self, *args):                  
        # Se crea el circulo amarillo y se añade
        self.circulo = Ellipse(pos=(self.center_x - 50, 50), size=(100, 100))
        with self.layout.canvas:
            Color(1, 1, 0)  
            self.circulo_instr = InstructionGroup()

            self.circulo_instr.add(self.circulo)
            # Comprobamos que no haya otro circulo
            if self.circulo_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.circulo_instr)
        
        # Schedule the update of the image box every 1/30 seconds
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)


    # Funcion para el boton continuar
    def on_continuar(self, *args):
        self.controlador.on_continuar_calibracion()

    # Funcion para el boton inicio
    def on_inicio(self, *args):
        self.controlador.cambiar_estado_calibracion(0)
        self.controlador.change_screen('inicio')

        # Se detiene la camara y se limpia el circulo
        self.circulo_instr.clear()



    # Funcion para actualizar la vista
    def update_view(self, n):
        self.texto_explicativo.text = self.textos_calibracion[n]
        if n != 0:
            self.circulo_instr.clear()
            self.image_box.opacity = 0  # Hide the image box
        else:
            self.image_box.opacity = 1  # Show the image box
    
    def update_image_box(self, dt):
        # Only update the image box in calibration state 0
        if self.controlador.obtener_estado_cal() == 0:
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
        Clock.unschedule(self.update_image_box)
