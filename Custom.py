from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.properties import ListProperty
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.properties import StringProperty
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
import os
from kivy.uix.widget import Widget


Builder.load_file('kivy/custom.kv')

class ButtonRnd(Button):
    size_hint = ListProperty([1, None])

class CustomSpinner(Spinner):
    pass

class CasillaTablero(Button):
    pass

class CasillaTableroPicto(ButtonBehavior, BoxLayout):
    source = StringProperty('')
    text = StringProperty('')

    def __init__(self, **kwargs):
        super(CasillaTableroPicto, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.label = Label(text = self.text, size_hint_y=None, height=40, halign='center', valign='middle', font_name='Texto', font_size=32, color=(0, 0, 0, 1))
        #Comprobar la existencia de la imagen
        if not os.path.isfile(self.source):
            self.source = './tableros/pictogramas/NOFOTO.png'

        self.image = AsyncImage(source=self.source, allow_stretch=False, keep_ratio=True)
        self.add_widget(self.image)
        self.add_widget(Widget(size_hint_y=0.2))
        self.add_widget(self.label)
        self.add_widget(Widget(size_hint_y=0.03))
