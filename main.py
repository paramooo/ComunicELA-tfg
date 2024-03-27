from kivy.app import App
from Modelo import Modelo
from Vista import Vista
from Controlador import Controlador
from kivy.config import Config

Config.set('graphics', 'fullscreen', 'auto')
Config.set('input', 'mouse', 'mouse,disable_multitouch')

class MyApp(App):
    def build(self):
        self.modelo = Modelo()
        vista = Vista()
        controlador = Controlador(self.modelo, vista)
        vista.set_controlador(controlador)
        vista.crear_pantallas()

        return vista.sm
    
    def on_stop(self):
            # Detiene la cámara antes de cerrar la aplicación
            self.modelo.detener_camara()

if __name__ == '__main__':
    MyApp().run()
