from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd, CustomSpinner
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class Inicio(Screen):
    def __init__(self, controlador, **kwargs):
        super(Inicio, self).__init__(**kwargs)
        self.controlador = controlador
        #self.background_color = (0, 0, 0, 1)
        self.porcentajeDisp = 0.15

        self.primera = True
        self.texto_inicio = ("Bienvenido a ComunicELA, una aplicación en desarrollo para ayudar a personas a comunicarse.\n" + 
                            "Para empezar, seleccione la cámara que quiera usar, despues, comienza calibrando el\n" + 
                            "parpadeo para que la aplicacion se ajuste a tu perfil.\n" +
                            "Una vez calibrado, puedes realizar un test para comprobar que todo funciona correctamente.\n" +
                            "Si todo va bien, puedes empezar a recopilar datos, muchas gracias!\n" + 
                            "Presione ESC en cuaquier momento para cerrar la aplicación")


        
        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)
        
        # Alineamos horizontalmente para separar en dos
        Principal = BoxLayout(orientation='vertical', padding=20)

        titulo = Label(text='ComunicELA', font_size='100sp', halign='center', color=(1, 1, 1, 1), size_hint=(1, 0.2), font_name='Titulo')
        
        caja = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))

        # Parte izquierda con los botones y el titulo
        Izquierda = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=20)

        # Menu de seleccion de camara
        self.camera_spinner = CustomSpinner(
            text='Cargando cámaras...',
            values=[],
            size_hint=(1, 0.1),
            pos_hint={'center_x': 0.5},)

        btn_cal = ButtonRnd(text='Calibrar parpadeo', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('calibrar'), font_name='Texto')

        btn_tst = ButtonRnd(text='Test', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('test'), font_name='Texto')

        btn_rec = ButtonRnd(text='Recopilar', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_r('recopilar') , font_name='Texto')

        btn_ree = ButtonRnd(text='Reentrenar', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('reentrenar'), font_name='Texto')

        btn_tab = ButtonRnd(text='Tableros', size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('tableros'), font_name='Texto')
        
        self.camera_spinner.bind(text=self.seleccionar_camara)
        espacio_blanco2 = BoxLayout(size_hint=(1, 0.05))

        # Parte derecha con el texto y la imagen
        Derecha = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=10)
       
        self.image_box = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=True)

        texto = Label(text=self.texto_inicio, halign='center', font_size=self.controlador.get_font_txts(), valign='center', color=(1, 1, 1, 1), font_name='Texto')



        # Montamos la estructura
        Izquierda.add_widget(self.camera_spinner)      
        Izquierda.add_widget(btn_cal)
        Izquierda.add_widget(btn_ree)
        Izquierda.add_widget(btn_tab)
        Izquierda.add_widget(btn_tst)
        Izquierda.add_widget(btn_rec)
        Izquierda.add_widget(espacio_blanco2)

        Derecha.add_widget(self.image_box)
        Derecha.add_widget(texto)

        caja.add_widget(Izquierda)
        caja.add_widget(Derecha)

        Principal.add_widget(titulo)
        Principal.add_widget(caja)

        self.add_widget(Principal)
        


    def on_enter(self, *args):
        #Menu de seleccion de camara una vez dentro para asi poder actualizar las camaras
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
        frame = self.controlador.get_frame_editado(self.porcentajeDisp)
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
