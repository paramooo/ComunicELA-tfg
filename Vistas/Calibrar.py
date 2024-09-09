from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from KivyCustom.Custom import ButtonRnd
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, Ellipse
from kivy.graphics import InstructionGroup
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.uix.widget import Widget
from ajustes.utils import get_recurso


class Calibrar(Screen):
    """
    Pantalla de calibración de la aplicación.
    """
    def __init__(self, controlador, **kwargs):
        super(Calibrar, self).__init__(**kwargs)
        self.controlador = controlador

        # Crear una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo2()  )
        self.add_widget(self.fondo)

        # Textos de calibración
        self.textos_calibracion = [
            self.controlador.get_string('mensaje_calibracion_1'),
            self.controlador.get_string('mensaje_calibracion_2'),
            self.controlador.get_string('mensaje_calibracion_3'),
            self.controlador.get_string('mensaje_calibracion_4'),
        ]

        # Crear el contenedor principal
        self.layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.add_widget(self.layout)

        # Crear el layout del menú
        self.menu_layout = BoxLayout(orientation='horizontal', padding=20, spacing=20, size_hint=(0.9, 0.6), pos_hint={'center_x': 0.5})
        self.layout.add_widget(self.menu_layout)

        # Botón de Continuar
        self.btn_comenzar = ButtonRnd(text=self.controlador.get_string('continuar'), size_hint=(0.25, 0.05), pos_hint={'center_x': 0.5}, on_press=self.on_continuar, font_name='Texto')
        self.layout.add_widget(self.btn_comenzar)
        self.layout.add_widget(Widget(size_hint_y=0.1))

        # Secciones
        left_section = BoxLayout(orientation='vertical', size_hint=(0.45, 1))
        right_section = BoxLayout(orientation='vertical', size_hint=(0.45, 1))
        self.menu_layout.add_widget(left_section)
        self.menu_layout.add_widget(right_section)

        # Botón de Inicio
        left_section.add_widget(Widget(size_hint_y=0.6))
        self.btn_inicio = ButtonRnd(text=self.controlador.get_string('inicio'), size_hint=(0.4, 0.28), pos_hint={'x': 0.05}, on_press=self.on_inicio, font_name='Texto')
        left_section.add_widget(self.btn_inicio)
        left_section.add_widget(Widget(size_hint_y=0.3))

        # Foto calibrar.png
        self.image = Image(source=get_recurso('imagenes/calibrar0.png'), pos_hint={'center_x': 0.5}, size_hint=(1.8,1.8))
        left_section.add_widget(self.image)

        # Texto explicativo
        self.texto_explicativo = Label(
            text=self.textos_calibracion[0],
            font_size='30',
            font_name='Texto',
            halign='center',
            valign='middle',
            color=(0,0,0,1), 
        )
        self.texto_explicativo.bind(size=self.texto_explicativo.setter('text_size'))
        left_section.add_widget(self.texto_explicativo)

        # Sección derecha
        right_section.add_widget(Widget(size_hint_y=0.15))
        self.image_box = Image(size_hint=(1, 0.7), pos_hint={'center_x': 0.5})
        right_section.add_widget(self.image_box)

        # Círculo calibración
        self.circulo = Ellipse(pos=(self.center_x - 30, 30), size=(60, 60))
        self.circulo_instr = InstructionGroup()

        # Programar la actualización de la image_box
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)


    def on_enter(self, *args):
        """
        Acciones a realizar al entrar en la pantalla de calibración.
        """
        self.update_idioma()
        self.update_divisoria()
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)
        self.update_view(self.controlador.cambiar_estado_calibracion(0))


    def update_divisoria(self):
        """
        Actualiza la divisoria de la pantalla para ajustarla al tamaño de la pantalla.
        """
        with self.layout.canvas:
            Color(1, 1, 1)
            self.divisoria = Rectangle(size=(2, self.height / 1.7), pos=(self.center_x, self.center_y - self.height / 4))
            self.divisoria_instr = InstructionGroup()
            self.divisoria_instr.add(self.divisoria)
            if self.divisoria_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.divisoria_instr)


    def on_continuar(self, *args):
        """
        Se acciona al pulsar el botón de continuar, cambia el estado de la calibración.
        """
        menu = self.controlador.cambiar_estado_calibracion()
        if(menu == 0):
            self.controlador.change_screen('inicio')
        elif menu != None:
            self.update_view(menu)


    def on_inicio(self, *args):
        """
        Vuelve al inicio de la aplicación, se acciona al pulsar el botón de inicio.
        """
        self.controlador.cambiar_estado_calibracion(0)
        self.controlador.change_screen('inicio')
        self.circulo_instr.clear()


    def update_view(self, estado):
        """
        Establece el texto y la imagen de la pantalla de calibración según el estado.

        Args:
            estado (int): Estado de la calibración
        """
        self.texto_explicativo.text = self.controlador.get_string(f'mensaje_calibracion_{estado+1}')
        self.image.source = get_recurso(f'imagenes/calibrar{estado}.png')

        if estado == 1:
            # Calcular la posición central
            center_x = self.width / 2 - self.circulo.size[0] / 2
            self.circulo.pos = (center_x, 10)

            with self.layout.canvas:
                Color(1, 1, 0)
                self.circulo_instr.clear()
                self.circulo_instr.add(Color(1, 1, 0))
                self.circulo_instr.add(self.circulo)
                if self.circulo_instr not in self.layout.canvas.children:
                    self.layout.canvas.add(self.circulo_instr)
        else:
            self.circulo_instr.clear()

        self.update_divisoria()


    def update_image_box(self, dt):
        """
        Actualiza la imagen de la cámara en la pantalla de calibración.
        """
        frame = self.controlador.get_frame_editado()
        if frame is None:
            return

        # Convert the frame to a texture
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')

        # Invertir la imagen verticalmente
        texture.flip_vertical()
        self.image_box.texture = texture


    def on_leave(self, *args):
        """
        Al salir de la pantalla de calibración, se detiene la actualización de la imagen de la cámara.
        """
        Clock.unschedule(self.update_image_box)

    def update_idioma(self):
        """
        Al actualizar el idioma cambia los textos correspondientes.
        """
        self.btn_comenzar.text = self.controlador.get_string('continuar')
        self.texto_explicativo.text = self.controlador.get_string(f'mensaje_calibracion_1')
        self.btn_inicio.text = self.controlador.get_string('inicio')