from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from Custom import ButtonRnd

class Inicio(Screen):
    def __init__(self, controlador, **kwargs):
        super(Inicio, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1)

        self.texto_inicio = ("Bienvenido a ComunicELA, una aplicación en desarrollo para ayudar a personas con ELA.\n" + 
                            "Para empezar, comienza calibrando el parpadeo para que la aplicacion se ajuste a tu perfil.\n" +
                            "Una vez calibrado, puedes realizar un test para comprobar que todo funciona correctamente.\n" +
                            "Si todo va bien, puedes empezar a recopilar datos, muchas gracias!\n" + 
                            "Presione ESC en cuaquier momento para cerrar la aplicación")

        # Alineamos horizontalmente para separar en dos
        Principal = BoxLayout(orientation='horizontal', padding=20)

        # Parte izquierda con los botones y el titulo
        Izquierda = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=10)

        titulo = Label(text='ComunicELA', font_size='100sp', halign='center', color=(1, 1, 1, 1))

        btn1 = ButtonRnd(text='Calibrar parpadeo', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('calibrar'))

        btn2 = ButtonRnd(text='Test', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('test'))

        btn3 = ButtonRnd(text='Recopilar', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen_r('recopilar'))

        btn4 = ButtonRnd(text='Reentrenar', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('reentrenar'))

        btn5 = ButtonRnd(text='Tableros', size_hint=(1, 0.5), on_press=lambda x: self.controlador.change_screen('tableros'))

        # Parte derecha con el texto y la imagen
        Derecha = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=10)

        texto = Label(text=self.texto_inicio, halign='center', font_size=self.controlador.get_font_txts(), valign='center', color=(1, 1, 1, 1))
        imagen = Image(source='imagenes/fic.png')

        # Montamos la estructura
        Izquierda.add_widget(titulo)
        Izquierda.add_widget(btn1)
        Izquierda.add_widget(btn2)
        Izquierda.add_widget(btn3)
        Izquierda.add_widget(btn4)
        Izquierda.add_widget(btn5)

        Derecha.add_widget(texto)
        Derecha.add_widget(imagen)

        Principal.add_widget(Izquierda)
        Principal.add_widget(Derecha)

        self.add_widget(Principal)
