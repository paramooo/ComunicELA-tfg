from kivy.uix.screenmanager import ScreenManager
from Inicio import Inicio
from Calibrar import Calibrar
from Test import Test
from Tableros import Tableros
from Recopilar import Recopilar
from Reentrenar import Reentrenar
from TablerosInstruc import TablerosInstruc


class Vista:
    def __init__(self):
        self.controlador = None
        self.sm = ScreenManager()

        self.inicio = None
        self.calibrar = None
        self.test = None
        self.tableros = None
        self.fondo = './imagenes/fondo_menus.jpg'
        self.fondo2 = './imagenes/fondo_menus2.jpg'

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

        self.reentrenar = Reentrenar(self.controlador, name='reentrenar')
        self.sm.add_widget(self.reentrenar)

        self.tabinstruc = TablerosInstruc(self.controlador, name='tabinstruc')
        self.sm.add_widget(self.tabinstruc)

    def change_screen(self, nombre):
        self.sm.current = nombre

    def get_screen(self):
        return self.sm.current 
    
    def get_fondo(self):
        return self.fondo
    
    def get_fondo2(self):
        return self.fondo2 