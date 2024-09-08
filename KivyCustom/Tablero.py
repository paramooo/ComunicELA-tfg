from kivy.uix.gridlayout import GridLayout
from KivyCustom.Custom import CasillaTablero
from KivyCustom.Custom import CasillaTableroPicto
from ajustes.utils import get_recurso

class Tablero(GridLayout):

    """
    Esta clase crea un tablero con las palabras y las imágenes correspondientes
    Patron de diseño: Composite
    """

    def __init__(self, palabras_con_imagenes, controlador, pictos, **kwargs):
        super(Tablero, self).__init__(**kwargs)
        self.controlador = controlador
        self.rows = len(palabras_con_imagenes)
        self.cols = len(palabras_con_imagenes[0]) if palabras_con_imagenes else 0
        self.spacing = [10, 10]
        self.casillas = []
        self.pictos = pictos
        for fila in palabras_con_imagenes:
            for imagen, palabra in fila:
                if pictos:
                    btn = CasillaTableroPicto(text=str(palabra), source=get_recurso(f'tableros/pictogramas/{imagen}'), on_press=self.on_button_press)
                else:
                    btn = CasillaTablero(text=str(palabra), on_press=self.on_button_press, font_name='Texto')
                self.casillas.append(btn)
                self.add_widget(btn)

    # Función que se ejecuta al pulsar una casilla
    def on_button_press(self, instance):
        if self.pictos:
            self.controlador.on_casilla_tab(instance.label.text)
        else:
            self.controlador.on_casilla_tab(instance.text)