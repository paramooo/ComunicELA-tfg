from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.properties import ListProperty
from kivy.uix.spinner import Spinner
from kivy.properties import StringProperty
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from os.path import isfile as os_isfile
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from ajustes.utils import get_recurso

Builder.load_file(get_recurso('kivyCustom/custom.kv'))
class HoverBehavior(object):
    """Hover behavior.
    :Events:
        `on_enter`
            Fired when mouse enter the widget.
        `on_leave`
            Fired when mouse leave the widget.
    """

    def __init__(self, **kwargs):
        self.hovered = False
        self.register_event_type('on_enter')
        self.register_event_type('on_leave')
        super(HoverBehavior, self).__init__(**kwargs)
        Window.bind(mouse_pos=self._on_mouse_pos)

    def _on_mouse_pos(self, *args):
        pos = args[1]
        # Check if mouse is inside the widget
        inside = self.collide_point(*self.to_widget(*pos))
        visible = self.get_root_window() is not None
        if self.hovered == inside or not visible:              
            return
        self.hovered = inside
        if inside:
            Window.set_system_cursor('hand')
            self.dispatch('on_enter')
        else:
            Window.set_system_cursor('arrow')
            self.dispatch('on_leave')

    def on_press(self):
        Window.set_system_cursor('arrow')
        Button.on_press(self)

    def on_enter(self):
        pass

    def on_leave(self):
        pass

class ButtonRnd(HoverBehavior, Button):
    size_hint = ListProperty([1, None])
    
class CustomSpinner(HoverBehavior, Spinner):
    pass

class CustomTextInput(TextInput):
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.background_color = (0.3, 0.3, 0.3, 1)
            return True
        return super(CustomTextInput, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            self.background_color = (0.7, 0.7, 0.7, 1)
            return True
        return super(CustomTextInput, self).on_touch_up(touch)
    
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
        if not os_isfile(self.source):
            self.source = get_recurso('imagenes/NOFOTO.png')

        self.image = AsyncImage(source=self.source )
        self.add_widget(self.image)
        self.add_widget(Widget(size_hint_y=0.2))
        self.add_widget(self.label)
        self.add_widget(Widget(size_hint_y=0.03))
