from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from KivyCustom.Custom import ButtonRnd
from ajustes.utils import get_recurso


class ReentrenarInstruc(Screen):
    """
    Pantalla que muestra las instrucciones para reentrenar el modelo.
    """
    def __init__(self, controlador, **kwargs):
        super(ReentrenarInstruc, self).__init__(**kwargs)
        self.controlador = controlador

        # Crear una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo2()  )
        self.add_widget(self.fondo)

        # Crear el contenedor principal
        self.layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        self.add_widget(self.layout)

        # Título de Instrucciones
        titulo_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.1))
        self.titulo = Label(
            text=self.controlador.get_string('instrucciones'), 
            font_size='50sp', 
            font_name='Texto',  # Usar tu fuente personalizada
            halign='center', 
            color=(1, 1, 1, 1),
            size_hint=(1, 1)
        )
        titulo_layout.add_widget(self.titulo)
        self.layout.add_widget(titulo_layout)

        # Layout para las instrucciones
        instrucciones_layout = BoxLayout(padding=20, spacing=20, size_hint=(1, 0.8), pos_hint={'center_x': 0.5})
        self.layout.add_widget(instrucciones_layout)

        # Instrucciones a la izquierda
        left_instructions = BoxLayout(orientation='vertical', size_hint=(0.6, 1))

        # Botón de Inicio
        self.btn_inicio = ButtonRnd(text=self.controlador.get_string('inicio'), size_hint=(0.3, 0.4), pos_hint={'x': 0.13}, on_press=self.on_inicio, font_name='Texto')
        left_instructions.add_widget(self.btn_inicio)
        instrucciones_layout.add_widget(left_instructions)

        #Instrucciones
        self.instruccion1 = Label(
            text=self.controlador.get_string('instruccion_re1'),
            font_size='24sp',
            halign='center',
            valign='middle',
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion1)
        self.instruccion2 = Label(
            text=self.controlador.get_string('instruccion_re2'),
            font_size='24sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion2)

        self.instruccion3 = Label(
            text=self.controlador.get_string('instruccion_re3'),
            font_size='24sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion3)

        #ESTO ES PORQUE SE AJUSTA DEMASIADO LA RED AL USUARIO DEBO REVISAR COMO HACER LOS REENTRENOS
        self.instruccion4 = Label(
            text=self.controlador.get_string('instruccion_re4'),
            font_size='24sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion4)

        self.instruccion5 = Label(
            text=self.controlador.get_string('instruccion_re5'),
            font_size='24sp',
            halign='center',
            valign='middle', 
            pos_hint={'x': 0.08}, 
            color=(0,0,0,1),
            font_name='Texto'
        )
        left_instructions.add_widget(self.instruccion5)

        # Botón de Descartar Reentrenamiento
        self.btn_descartar = ButtonRnd(text=self.controlador.get_string('descartar_ree'), size_hint=(0.4, 0.3), pos_hint={'center_x': 0.6}, on_press=self.on_descartar, font_name='Texto')
        left_instructions.add_widget(self.btn_descartar)

        left_instructions.add_widget(Widget(size_hint_y=0.8))

        # Botones a la derecha
        right_buttons = BoxLayout(orientation='vertical', size_hint=(0.4, 1))

        #Gif de reentrenamiento
        right_buttons.add_widget(Widget(size_hint_y=0.15))
        imagen1 = Image(source=get_recurso('imagenes/reentrenargif.zip'), size_hint=(0.6, 0.25), anim_delay=0.06, pos_hint={'center_x': 0.5})
        right_buttons.add_widget(imagen1)

        # Botón de Reentrenar
        self.btn_reentrenar = ButtonRnd(text=self.controlador.get_string('reentrenar'), size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5}, on_press=self.on_reentrenar, font_name='Texto')
        right_buttons.add_widget(self.btn_reentrenar)
        right_buttons.add_widget(Widget(size_hint_y=0.25))

        instrucciones_layout.add_widget(right_buttons)

    def on_enter(self, *args):
        """
        Al entrar en la pantalla, se actualiza el idioma de los elementos       
        """
        self.update_idioma()


    def on_descartar(self, *args):
        """
        Descarta los reentrenamientos realizados
        """
        self.controlador.descartar_reentrenamientos()

    def on_reentrenar(self, *args):
        """
        Abre la pantalla de reentrenamiento, la de recopilar los datos
        """
        self.controlador.change_screen('reentrenar')
    
    def on_inicio(self, *args):
        """
        Abre la pantalla de inicio
        """
        self.controlador.change_screen('inicio')


    def update_idioma(self):
        """
        Actualiza el idioma de los elementos de la pantalla
        """
        self.btn_descartar.text = self.controlador.get_string('descartar_ree')
        self.btn_reentrenar.text = self.controlador.get_string('reentrenar')
        self.btn_inicio.text = self.controlador.get_string('inicio')
        self.instruccion1.text = self.controlador.get_string('instruccion_re1')
        self.instruccion2.text = self.controlador.get_string('instruccion_re2')
        self.instruccion3.text = self.controlador.get_string('instruccion_re3')
        self.instruccion4.text = self.controlador.get_string('instruccion_re4')
        self.instruccion5.text = self.controlador.get_string('instruccion_re5')
        self.titulo.text = self.controlador.get_string('instrucciones')
