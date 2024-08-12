from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox
from KivyCustom.Custom import ButtonRnd
from KivyCustom.PopUp import CustPopup
from kivy.graphics import Line

class PopUpAjustes(ModalView):
    def __init__(self, camera_spinner, voz_spinner, boton_gemini,show_tutorial, controlador, **kwargs):
        super(PopUpAjustes, self).__init__(**kwargs)
        self.controlador = controlador
        self.show_tutorial = show_tutorial
        self.size_hint = (0.5, 0.5)  # Tamaño del popup

        # Crea un layout con un fondo semi-transparente
        layout = BoxLayout(orientation='vertical', padding=[20,20,20,20], spacing=10)
        with layout.canvas.before:
            Color(0.65, 0.65, 0.65, 1) 
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
            Color(0, 0, 0, 1)  
            self.line = Line(rectangle=(layout.x, layout.y, layout.width, layout.height), width=2)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # Añade un título al layout -> 0.2 de alto
        self.label_titulo = Label(text=self.controlador.get_string('ajustes'), font_size='30sp', font_name='Texto', halign='center', color=(0, 0, 0, 1), size_hint=(1, 0.2))
        layout.add_widget(self.label_titulo)
        layout.add_widget(Widget(size_hint=(1, 0.05)))

        # Layout para los spinners -> 0.3 de alto
        layout_botones = BoxLayout(orientation='horizontal', size_hint=(1, 0.3))
        # Añade los spinners y el botón al layout

        layout_camara = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        self.label_select_camera = Label(text=self.controlador.get_string('seleccion_camara'), font_name='Texto', color=(0, 0, 0, 1), font_size='20sp')
        layout_camara.add_widget(self.label_select_camera)
        layout_camara.add_widget(camera_spinner)

        layout_voz = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        self.label_select_voz = Label(text=self.controlador.get_string('seleccion_voz'), font_name='Texto', color=(0, 0, 0, 1), font_size='20sp')
        layout_voz.add_widget(self.label_select_voz)  
        layout_voz.add_widget(voz_spinner)

        layout_botones.add_widget(layout_camara)
        layout_botones.add_widget(Widget(size_hint=(0.07, 1)))
        layout_botones.add_widget(layout_voz)
        layout.add_widget(layout_botones)
        layout.add_widget(Widget(size_hint=(1, 0.1)))

        layout_gemini = BoxLayout(orientation='vertical', size_hint=(1, 0.4))
        layout_gemini.add_widget(Widget(size_hint=(1, 0.05)))
        self.label_gemini = Label(text=self.controlador.get_string('conjugar'), font_name='Texto', color=(0, 0, 0, 1), font_size='20sp', size_hint=(1, 0.1))
        layout_gemini.add_widget(self.label_gemini)

        layout_btns_gemini = BoxLayout(orientation='horizontal', size_hint=(0.5, 0.05), pos_hint={'center_x': 0.5})
        layout_btns_gemini.add_widget(boton_gemini)
        layout_btns_gemini.add_widget(Widget(size_hint=(0.01, 0.3)))
        boton_info = ButtonRnd(text="?", size_hint=(0.1, 1), font_name='Texto', font_size='25sp', on_release=self.on_info)
        layout_btns_gemini.add_widget(boton_info)
        layout_gemini.add_widget(layout_btns_gemini)
        layout.add_widget(layout_gemini)
        layout.add_widget(Widget(size_hint=(1, 0.1)))

        # Añade un botón para cerrar el popup
        self.fondo_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.close_button = ButtonRnd(text=self.controlador.get_string('cerrar'), size_hint=(0.15, 1), on_release=self.dismiss, font_name='Texto', font_size='18sp', pos_hint={'center_x': 0.9})
        self.tutorial_button = ButtonRnd(text=self.controlador.get_string('tutorial'), size_hint=(0.15, 1), on_release=self.on_tutorial, font_name='Texto', font_size='18sp', pos_hint={'center_x': 0.1})
        self.fondo_layout.add_widget(self.tutorial_button)
        self.fondo_layout.add_widget(Widget(size_hint=(0.7, 1)))
        self.fondo_layout.add_widget(self.close_button)

        layout.add_widget(self.fondo_layout)        

        self.add_widget(layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        self.line.rectangle = (instance.x, instance.y, instance.width, instance.height)

    def on_info(self, instance):
        CustPopup(self.controlador.get_string('info_gemini'), func_continuar=self.on_dismiss, pos = (0.5, 0.5), controlador=self.controlador).open()

    def on_tutorial(self, instance):
        self.dismiss()
        self.show_tutorial()
