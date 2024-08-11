from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics import Color, Line, Rectangle, InstructionGroup
from KivyCustom.Custom import ButtonRnd
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
            text=self.controlador.get_string('instrucciones'), 
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
        self.btn_inicio = ButtonRnd(text=self.controlador.get_string('inicio'), size_hint=(0.37, 0.35), pos_hint={'x': 0.15}, on_press=self.on_inicio, font_name='Texto')
        left_instructions.add_widget(self.btn_inicio)
        left_instructions.add_widget(Widget(size_hint_y=0.10))
        
        self.instruccion1 = Label(
            text=self.controlador.get_string('instruccion_tab1'),
            font_size='25sp',
            halign='center',
            valign='middle',
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion1)
        self.instruccion2 = Label(
            text=self.controlador.get_string('instruccion_tab2'),
            font_size='25sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion2)
        self.instruccion3 = Label(
            text=self.controlador.get_string('instruccion_tab3'),
            font_size='25sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion3)
        self.instruccion4 = Label(
            text=self.controlador.get_string('instruccion_tab4'),
            font_size='25sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion4)

        self.instruccion5 = Label(
            text=self.controlador.get_string('instruccion_tab5'),
            font_size='25sp',
            halign='center',
            valign='middle',
            pos_hint={'x': 0.08},
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion5)

        left_instructions.add_widget(Widget(size_hint_y=0.8))

        instrucciones_layout.add_widget(left_instructions)

        # Instrucciones a la derecha
        right_instructions = BoxLayout(orientation='vertical', size_hint=(0.5, 1))

        # Imagen 1
        imagen1 = Image(source='./imagenes/tabpicto.png', size_hint=(0.6, 0.25), pos_hint={'center_x': 0.5})
        right_instructions.add_widget(imagen1)

        # Botón de Comenzar
        self.btn_comenzar = ButtonRnd(text=self.controlador.get_string('comenzar_con_pic'), size_hint=(0.6, 0.05), pos_hint={'center_x': 0.5}, on_press=self.on_comenzar, font_name='Texto')
        right_instructions.add_widget(self.btn_comenzar)
        right_instructions.add_widget(Widget(size_hint_y=0.05))

        # Imagen 2
        imagen2 = Image(source='./imagenes/tabletras.png', size_hint=(0.6, 0.25), pos_hint={'center_x': 0.5})
        right_instructions.add_widget(imagen2)

        # Botón de Comenzar2
        self.btn_comenzar2 = ButtonRnd(text=self.controlador.get_string('comenzar_con_texto'), size_hint=(0.6, 0.05), pos_hint={'center_x': 0.5}, on_press=self.on_comenzar2, font_name='Texto')
        right_instructions.add_widget(self.btn_comenzar2)
        right_instructions.add_widget(Widget(size_hint_y=0.1))

        instrucciones_layout.add_widget(right_instructions)

    # Función para el botón de Inicio
    def on_inicio(self, *args):
        self.controlador.change_screen('inicio')

    # Función para el botón de Comenzar
    def on_comenzar(self, *args):
        self.controlador.set_pictogramas(True)
        self.controlador.change_screen('tableros')

    # Función para el botón de Comenzar2
    def on_comenzar2(self, *args):
        self.controlador.set_pictogramas(False)
        self.controlador.change_screen('tableros')

    def on_pre_enter(self, *args):
        self.actualizar_idioma()

    def actualizar_idioma(self):
        self.btn_inicio.text = self.controlador.get_string('inicio')
        self.btn_comenzar.text = self.controlador.get_string('comenzar_con_pic')
        self.btn_comenzar2.text = self.controlador.get_string('comenzar_con_texto')
        self.instruccion1.text = self.controlador.get_string('instruccion_tab1')
        self.instruccion2.text = self.controlador.get_string('instruccion_tab2')
        self.instruccion3.text = self.controlador.get_string('instruccion_tab3')
        self.instruccion4.text = self.controlador.get_string('instruccion_tab4')
        self.instruccion5.text = self.controlador.get_string('instruccion_tab5')
