from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.modalview import ModalView
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox


class CustPopup(ModalView):
    def __init__(self, message, func_continuar, pos, controlador, show_switch=False, func_volver=None, func_saltar=None, **kwargs):
        super(CustPopup, self).__init__(**kwargs)
        self.controlador = controlador
        self.size_hint = (0.3, 0.2)  # Tamaño del popup
        self.auto_dismiss = False  # No permitir que se cierre al pulsar fuera
        self.pos_hint = {'center_x': pos[0], 'center_y': pos[1]}  # Posición del popup

        # Funciones a ejecutar al pulsar los botones
        self.func_volver = func_volver
        self.func_saltar = func_saltar
        self.func_continuar = func_continuar

        # Crea un layout con un fondo semi-transparente
        layout = BoxLayout(orientation='vertical', padding=[10, 10, 10, 10])
        with layout.canvas.before:
            Color(0, 0, 0, 0.7)  # color negro con alpha a 0.7
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # Añade un Label con el mensaje del tutorial
        label = Label(text=message, font_size=20, halign='center', valign='middle')
        label.bind(size=label.setter('text_size'))  # Para que el texto se ajuste al tamaño del Label
        layout.add_widget(label)

        if show_switch:
            checkbox_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
            checkbox_label = Label(text='No volver a mostrar el tutorial', color=(1, 1, 1, 1))
            self.tutorial_checkbox = CheckBox(active=False)
            checkbox_layout.add_widget(Widget(size_hint_x = 1))
            checkbox_layout.add_widget(checkbox_label)
            checkbox_layout.add_widget(self.tutorial_checkbox)
            checkbox_layout.add_widget(Widget(size_hint_x=0.5))
            layout.add_widget(checkbox_layout)
            layout.add_widget(Widget(size_hint_y = 0.05))

        # Añade un botón para cerrar el popup
        button = ButtonRnd(text='Continuar', size_hint=(1, 0.3), on_release=self.on_continuar)
        layout.add_widget(button)

        if func_volver:
            button_volver = ButtonRnd(text='Atrás', size_hint=(1, 0.3), on_release=self.on_volver)
            self.add_widget(Widget(size_hint_y=0.05))
            layout.add_widget(button_volver)
        
        if func_saltar:
            button_siguiente = ButtonRnd(text='Saltar', size_hint=(1, 0.3), on_release=self.on_saltar)
            self.add_widget(Widget(size_hint_y=0.05))
            layout.add_widget(button_siguiente)

        self.add_widget(layout)
        self.call_on_dismiss = True

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_continuar(self, instance):
        # Guarda la configuración cuando se cierre el popup
        if hasattr(self, 'tutorial_checkbox'):
            self.controlador.set_show_tutorial(not self.tutorial_checkbox.active)
        if self.func_continuar:
            self.func_continuar()
        self.dismiss()


    def on_saltar(self, instance):
        if self.func_saltar:
            self.func_saltar()
        self.dismiss()

    def on_volver(self, instance):
        if self.func_volver:
            self.func_volver()
        self.dismiss()