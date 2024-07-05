from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.switch import Switch
from Custom import ButtonRnd, CustomSpinner
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.modalview import ModalView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.widget import Widget


class TutorialPopup(ModalView):
    def __init__(self, message, on_dismiss, pos, **kwargs):
        super(TutorialPopup, self).__init__(**kwargs)
        self.size_hint = (0.3, 0.2)  # Tamaño del popup
        self.auto_dismiss = False  # No permitir que se cierre al pulsar fuera
        self.pos_hint = {'center_x': pos[0], 'center_y': pos[1]}  # Posición del popup

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

        # Añade un botón para cerrar el popup
        button = ButtonRnd(text='Continuar', size_hint=(1, 0.3), on_release=self.dismiss)
        layout.add_widget(button)

        self.add_widget(layout)
        self.bind(on_dismiss=on_dismiss)  # Función a llamar cuando se cierre el popup

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class Inicio(Screen):
    def __init__(self, controlador, **kwargs):
        super(Inicio, self).__init__(**kwargs)
        self.controlador = controlador
        self.primera = True

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)
        
        # Alineamos horizontalmente para separar en dos
        Principal = BoxLayout(orientation='vertical', padding=20)

        titulo = Label(text='ComunicELA', font_size='100sp', halign='center', color=(1, 1, 1, 1), size_hint=(1, 0.2), font_name='Titulo')
        
        caja = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))

        # Parte izquierda con los botones y el titulo
        self.Izquierda = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=20)

        # Añadir el switch para opciones de desarrollador
        switch_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        switch_layout.add_widget(Widget(size_hint_x=1))
        switch_label = Label(text='Opciones de desarrollador', color=(1, 1, 1, 1))
        self.dev_switch = Switch(active=False)
        self.dev_switch.bind(active=self.opciones_des)
        switch_layout.add_widget(switch_label)
        switch_layout.add_widget(self.dev_switch)

        # Menu de seleccion de camara
        self.camera_spinner = CustomSpinner(
            text='Cargando cámaras...',
            values=[],
            size_hint=(0.6, 0.1),
            pos_hint={'center_x': 0.5},)

        self.btn_cal = ButtonRnd(text='Calibrar parpadeo', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('calibrar'), font_name='Texto')

        self.btn_tst = ButtonRnd(text='Test', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('test'), font_name='Texto')

        self.btn_rec = ButtonRnd(text='Recopilar', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_r('recopilar') , font_name='Texto')

        self.btn_ree = ButtonRnd(text='Reentrenar', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('reentrenar'), font_name='Texto')

        self.btn_tab = ButtonRnd(text='Tableros', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('tabinstruc'), font_name='Texto')
        
        self.camera_spinner.bind(text=self.seleccionar_camara)
        espacio_blanco2 = BoxLayout(size_hint=(1, 0.05))

        # Parte derecha con el texto y la imagen
        Derecha = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=10)
       
        self.image_box = Image(size_hint=(0.7, 0.7), pos_hint={'center_x': 0.5}, allow_stretch=True, keep_ratio=True)

        # Montamos la estructura
        self.Izquierda.add_widget(self.btn_cal)
        self.Izquierda.add_widget(self.btn_ree)
        self.Izquierda.add_widget(self.btn_tab)
        self.Izquierda.add_widget(espacio_blanco2)


        Derecha.add_widget(self.image_box)
        Derecha.add_widget(self.camera_spinner)  
        Derecha.add_widget(Widget(size_hint_y=0.1))    
        Derecha.add_widget(switch_layout) 

        caja.add_widget(self.Izquierda)
        caja.add_widget(Derecha)

        Principal.add_widget(titulo)
        Principal.add_widget(caja)

        self.add_widget(Principal)

        self.tutorial_buttons = [
            (self.camera_spinner, 'Bienvenido a ComunicELA, primero debe seleccionar la cámara que desea utilizar'),
            (self.btn_cal, 'Despues debe calibrar el parpadeo para comenzar a usar la aplicación'),
            (self.btn_tab, 'Posteriormente ya podrá utilizar los tableros de comunicación'),
            (self.btn_ree, 'Si no está satisfecho con el rendimiento de la aplicación,\n puede reentrenar el modelo para que se ajuste mejor a sus necesidades'),
            (self.dev_switch, 'Este apartado habilita opciones para pruebas y desarrollo de la aplicación'),
        ]

        # Llamar al método show_tutorial después de que la vista inicial se haya completado
        Clock.schedule_once(self.show_tutorial, 2)

    def show_tutorial(self, *args):
        if self.tutorial_buttons:
            button, message = self.tutorial_buttons.pop(0)
            
            # Calcula la posición normalizada
            if button == self.camera_spinner:
                pos = 0.5, 0.22
            elif button == self.dev_switch:
                pos = 0.8, 0.1
            else:
                pos = (button.center_x / Window.width) + 0.4, button.center_y / Window.height
            
            popup = TutorialPopup(message, self.show_tutorial, pos)
            popup.open()


    def on_enter(self, *args):
        # Menu de seleccion de camara una vez dentro para asi poder actualizar las camaras
        if self.primera:
            self.controlador.obtener_camaras()
            self.primera = False
    
        # Schedule the update of the image box every 1/30 seconds
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)

    def seleccionar_camara(self, _, text):
        if text.startswith('Cámara '):
            # Extrae el número de la cámara del texto
            camara = int(text.split(' ')[1])
            self.controlador.seleccionar_camara(camara)
        elif text == 'Actualizar cámaras':
            self.controlador.obtener_camaras()
        elif text == 'Seleccionar cámara' or text == 'Cargando cámaras...':
            pass
        else:
            self.controlador.seleccionar_camara(int(text))
            self.camera_spinner.text = f'Cámara {text}'

    def update_image_box(self, dt):
        frame = self.controlador.get_frame_editado()
        if frame is None:
            return
        
        # Convert the frame to a texture
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')

        # Invertir la imagen verticalmente
        texture.flip_vertical()
        self.image_box.texture = texture

    def on_leave(self, *args):
        Clock.unschedule(self.update_image_box)

    def opciones_des(self, instance, value):
        if value:
            self.Izquierda.add_widget(self.btn_tst) 
            self.Izquierda.add_widget(self.btn_rec) 
            self.controlador.set_desarrollador(True)
        else:
            self.Izquierda.remove_widget(self.btn_tst)  
            self.Izquierda.remove_widget(self.btn_rec) 
            self.controlador.set_desarrollador(False)
