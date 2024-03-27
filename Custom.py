from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.properties import ListProperty

Builder.load_file('kivy/custom.kv')

class ButtonRnd(Button):
    size_hint = ListProperty([1, None])
