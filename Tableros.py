from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label

class Tableros(Screen):
    def __init__(self, controlador, **kwargs):
        super(Tableros, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 

        # Agregar texto en la parte superior
        label = Label(text='Por desarrollar', size_hint=(1, 0.1), font_size=self.controlador.get_font_txts(), pos_hint={'top': 1})
        self.add_widget(label)

        btn = ButtonRnd(text='Inicio', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('inicio'))
        self.add_widget(btn)
