from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from KivyCustom.Custom import ButtonRnd
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.modalview import ModalView
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox
from kivy.graphics import Line

class CustPopup(ModalView):
    def __init__(self, message, func_continuar, pos, controlador, show_switch=False, func_volver=None, func_saltar=None, bt_empez=False, **kwargs):
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
            Color(0.7, 0.7, 0.7, 1)  # color negro con alpha a 0.7
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            Color(0, 0, 0, 1)  # color de borde negro
            self.line = Line(rectangle=(layout.x, layout.y, layout.width, layout.height), width=2)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # Añade un Label con el mensaje del tutorial
        label = Label(text=message, font_size=24, halign='center', valign='middle', font_name='Texto', color=(0, 0, 0, 1))
        label.bind(size=label.setter('text_size'))  # Para que el texto se ajuste al tamaño del Label
        layout.add_widget(label)

        if show_switch:
            checkbox_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))
            checkbox_label = Label(text=self.controlador.get_string('no_volver_mostrar'), color=(0, 0, 0, 1), font_name='Texto')
            self.tutorial_checkbox = CheckBox(active=False, color=(0,0,0,1))
            checkbox_layout.add_widget(Widget(size_hint_x = 1))
            checkbox_layout.add_widget(checkbox_label)
            checkbox_layout.add_widget(self.tutorial_checkbox)
            checkbox_layout.add_widget(Widget(size_hint_x=0.5))
            layout.add_widget(checkbox_layout)
            layout.add_widget(Widget(size_hint_y = 0.05))

        # Añade un botón para cerrar el popup
        if bt_empez:
            button = ButtonRnd(text=self.controlador.get_string('comenzar'), size_hint=(1, 0.3), on_release=self.on_continuar, font_name='Texto')
        else:
            button = ButtonRnd(text=self.controlador.get_string('continuar'), size_hint=(1, 0.3), on_release=self.on_continuar,font_name='Texto')
        layout.add_widget(button)


        # PARTE PARA LAS PRUEBAS
        if func_volver or func_saltar:
            self.size_hint = (0.3, 0.35)  # Tamaño del popup
            #Espacio entre los botones
            layout.add_widget(Widget(size_hint_y=0.1))
            layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, 0.3))
            
            # Añade un botón para volver 
            if func_volver:
                button_volver = ButtonRnd(text=self.controlador.get_string('atras'), size_hint=(0.45, 1), on_release=self.on_volver, font_name='Texto')
            else:
                button_volver = ButtonRnd(text=self.controlador.get_string('atras'), size_hint=(0.45, 1), disabled=True, font_name='Texto')
            layout_botones.add_widget(button_volver)


            # Añade un botón para saltar            
            if func_saltar:
                button_siguiente = ButtonRnd(text=self.controlador.get_string('saltar'), size_hint=(0.45, 1), on_release=self.on_saltar, font_name='Texto')
            else:
                button_siguiente = ButtonRnd(text=self.controlador.get_string('saltar'), size_hint=(0.45, 1), disabled=True, font_name='Texto')
            layout_botones.add_widget(button_siguiente)

            layout.add_widget(layout_botones)

        self.add_widget(layout)
        self.call_on_dismiss = True

    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        self.line.rectangle = (instance.x, instance.y, instance.width, instance.height)

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