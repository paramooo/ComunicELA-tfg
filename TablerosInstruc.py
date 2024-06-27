from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics import Color, Line, Rectangle, InstructionGroup
from Custom import ButtonRnd
from kivy.uix.widget import Widget

class TablerosInstruc(Screen):
    def __init__(self, controlador, **kwargs):
        super(TablerosInstruc, self).__init__(**kwargs)
        self.controlador = controlador

        # Crear una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo2(), allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)

        # Crear el contenedor principal
        self.layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.add_widget(self.layout)

        # Título de Instrucciones
        titulo_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.1))
        titulo = Label(
            text='INSTRUCCIONES', 
            font_size='50sp', 
            font_name='Texto',  # Usar tu fuente personalizada
            halign='center', 
            color=(1, 1, 1, 1),
            size_hint=(1, 1)
        )
        titulo_layout.add_widget(titulo)
        self.layout.add_widget(titulo_layout)

        # Layout para las instrucciones
        instrucciones_layout = BoxLayout(padding=20, spacing=20, size_hint=(1, 0.8), pos_hint={'center_x': 0.5})
        self.layout.add_widget(instrucciones_layout)


        # Instrucciones a la izquierda
        left_instructions = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        
        # Botón de Inicio
        left_instructions.add_widget(Widget(size_hint_y=0.05))
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(0.35, 0.21), pos_hint={'x': 0.14}, on_press=self.on_inicio, font_name='Texto')
        left_instructions.add_widget(btn_inicio)
        left_instructions.add_widget(Widget(size_hint_y=0.2))

        instruccion1 = Label(
            text='El cursor debe seguir su mirada',
            font_size='20sp',
            halign='center',
            valign='middle'
        )
        left_instructions.add_widget(instruccion1)
        instruccion2 = Label(
            text='Mantener el cursor sobre una casilla para seleccionarla',
            font_size='20sp',
            halign='center',
            valign='middle'
        )
        left_instructions.add_widget(instruccion2)
        instrucciones_layout.add_widget(left_instructions)

        # Instrucciones a la derecha
        right_instructions = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        right_instructions.add_widget(Widget(size_hint_y=0.46))
        instruccion3 = Label(
            text='Parpadear para confirmar la selección',
            font_size='20sp',
            halign='center',
            valign='middle'
        )
        right_instructions.add_widget(instruccion3)
        instruccion4 = Label(
            text='Mantener los ojos cerrados 2 segundos para modo descanso',
            font_size='20sp',
            halign='center',
            valign='middle'
        )
        right_instructions.add_widget(instruccion4)
        instrucciones_layout.add_widget(right_instructions)

        # Botón de Continuar 
        btn_comenzar = ButtonRnd(text='Continuar', size_hint=(0.25, 0.07), pos_hint={'center_x': 0.5}, on_press=self.on_continuar, font_name='Texto')
        self.layout.add_widget(btn_comenzar)
        self.layout.add_widget(Widget(size_hint_y=0.15))

    # Función para el botón de Inicio
    def on_inicio(self, *args):
        self.controlador.change_screen('inicio')

    # Función para el botón de Continuar
    def on_continuar(self, *args):
        self.controlador.change_screen('tableros')
