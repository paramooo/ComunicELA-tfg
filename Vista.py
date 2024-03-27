from kivy.uix.screenmanager import ScreenManager
from Inicio import Inicio
from Calibrar import Calibrar
from Test import Test
from Tableros import Tableros
from Recopilar import Recopilar

class Vista:
    def __init__(self):
        self.controlador = None
        self.sm = ScreenManager()

        self.inicio = None
        self.calibrar = None
        self.test = None
        self.tableros = None

    def set_controlador(self, controlador):
        self.controlador = controlador

    def crear_pantallas(self):
        self.inicio = Inicio(self.controlador, name='inicio')
        self.sm.add_widget(self.inicio)

        self.calibrar = Calibrar(self.controlador, name='calibrar')
        self.sm.add_widget(self.calibrar)

        self.test = Test(self.controlador, name='test')
        self.sm.add_widget(self.test)

        self.tableros = Tableros(self.controlador, name='tableros')
        self.sm.add_widget(self.tableros)

        self.recopilar = Recopilar(self.controlador, name='recopilar')
        self.sm.add_widget(self.recopilar)


    def change_screen(self, nombre):
        self.sm.current = nombre
