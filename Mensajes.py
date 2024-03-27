from kivy.uix.label import Label
from kivy.animation import Animation

class Mensajes(Label):
    def __init__(self, mensaje, **kwargs):
        super().__init__(**kwargs)
        self.text = mensaje
        self.font_size = '25sp'  # Aumentar el tamaño de la fuente a 25
        self.color = (1, 1, 1, 1)  # Cambiar el color del texto a blanco
        self.background_color = (0, 0, 0, 1)  # Cambiar el color de fondo a negro
        self.pos_hint = {'center_x': 0.5, 'y': 0}
        self.size_hint = (None, None)
        self.size = (400, 50)

        anim = Animation(opacity=1, duration=0.5) + Animation(duration=2) + Animation(opacity=0, duration=0.5)
        anim.start(self)
