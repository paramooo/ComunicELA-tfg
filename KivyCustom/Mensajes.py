from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Rectangle, Color
from kivy.animation import Animation

class Mensajes(RelativeLayout):
    def __init__(self, mensaje, **kwargs):
        super().__init__(**kwargs)
        self.label = Label(text=mensaje, font_size='25sp', color=(1, 1, 1, 1), size_hint=(None, None))
        self.label.bind(texture_size=self._update_label_size)
        self.add_widget(self.label)

        with self.canvas.before:
            Color(0, 0, 0, 1)  # establece el color a negro
            self.rect = Rectangle(pos=self.label.pos, size=self.label.size)

        self.bind(pos=self._update_rect_pos, size=self._update_label_pos)
        self.label.bind(size=self._update_rect_size)

        anim = Animation(opacity=1, duration=0.5) + Animation(duration=2) + Animation(opacity=0, duration=0.5)
        anim.start(self.label)
        anim.start(self)

    def _update_label_size(self, instance, value):
        self.label.size = value
        self.label.text_size = value

    def _update_label_pos(self, instance, value):
        self.label.pos = (self.center_x - self.label.width / 2, 5)  # 5px más arriba
        self.rect.pos = (self.label.pos[0] - 5, self.label.pos[1] - 5)  # 5px más a la izquierda y abajo

    def _update_rect_pos(self, instance, value):
        self.rect.pos = self.label.pos

    def _update_rect_size(self, instance, value):
        self.rect.size = (self.label.width + 10, self.label.height + 10)  # 5px más ancho y alto
