from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.uix.image import Image

class Reentrenar(Screen):
    def __init__(self, controlador, **kwargs):
        super(Reentrenar, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)

        # Agregar texto en la parte superior
        label = Label(text='Por desarrollar', size_hint=(1, 0.1), font_size=self.controlador.get_font_txts(), pos_hint={'top': 1}, font_name='Texto')
        self.add_widget(label)

        btn = ButtonRnd(text='Inicio', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('inicio'), font_name='Texto')
        self.add_widget(btn)
