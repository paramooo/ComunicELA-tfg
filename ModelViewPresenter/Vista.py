from kivy.uix.screenmanager import ScreenManager
from Vistas.Inicio import Inicio
from Vistas.Calibrar import Calibrar
from Vistas.desarrollador.Test import Test
from Vistas.Tableros import Tableros
from Vistas.desarrollador.Recopilar import Recopilar
from Vistas.Reentrenar import Reentrenar
from Vistas.TablerosInstruc import TablerosInstruc
from Vistas.desarrollador.TablerosPruebas import TablerosPruebas
from Vistas.ReentrenarInstruc import ReentrenarInstruc
from ajustes.utils import get_recurso


class Vista:
    """
    Clase que se encarga de gestionar las pantallas de la aplicación.
    """
    def __init__(self):
        self.controlador = None
        self.sm = ScreenManager()

        self.inicio = None
        self.calibrar = None
        self.test = None
        self.tableros = None
        self.fondo = get_recurso('imagenes/fondo_menus.jpg')
        self.fondo2 = get_recurso('imagenes/fondo_menus3.jpg')

    def set_controlador(self, controlador):
        """
        Establece el presenter del MVP
        """
        self.controlador = controlador


    def crear_pantallas(self):
        """
        Crea las pantallas de la aplicación
        """
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

        self.reentrenarinstruc = ReentrenarInstruc(self.controlador, name='reentrenarinstruc')
        self.sm.add_widget(self.reentrenarinstruc)

        self.tabinstruc = TablerosInstruc(self.controlador, name='tabinstruc')
        self.sm.add_widget(self.tabinstruc)

        self.tablerosprueb = TablerosPruebas(self.controlador, name='tablerosprueb')
        self.sm.add_widget(self.tablerosprueb)


    def change_screen(self, nombre):
        """
        Cambia la pantalla actual por la que se le pasa por parámetro

        Args:
            nombre (str): Nombre de la pantalla a la que se quiere cambiar
        """
        self.sm.current = nombre


    def get_screen(self):
        """
        Devuelve la pantalla actual en la que se encuentra la aplicación
        """
        return self.sm.current 
    

    def get_fondo(self):
        """
        Devuelve el fondo 1 de la aplicación
        """
        return self.fondo
    

    def get_fondo2(self):
        """
        Devuelve el fondo 2 de la aplicación
        """
        return self.fondo2 