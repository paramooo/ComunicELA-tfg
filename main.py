from kivy.app import App
from ModelViewPresenter.Modelo import Modelo
from ModelViewPresenter.Vista import Vista
from ModelViewPresenter.Presenter import Presenter
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config

Window.fullscreen = 'auto'

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

class MyApp(App):
    def build(self):
        self.title = 'ComunicELA'  
        self.icon = './imagenes/logo.png'  
        LabelBase.register(name='Titulo', fn_regular='./KivyCustom/fuentes/Orbitron-Regular.ttf')
        LabelBase.register(name='Texto', fn_regular='./KivyCustom/fuentes/FrancoisOne-Regular.ttf')

        self.modelo = Modelo()
        vista = Vista()
        controlador = Presenter(self.modelo, vista)
        vista.set_controlador(controlador)
        vista.crear_pantallas()

        return vista.sm
    
    def on_stop(self):
            # Detiene la cámara antes de cerrar la aplicación
            self.modelo.detener_camara()

if __name__ == '__main__':
    MyApp().run()
