from kivy.uix.gridlayout import GridLayout
from KivyCustom.Custom import CasillaTablero
from KivyCustom.Custom import CasillaTableroPicto

class Tablero(GridLayout):
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
                    btn = CasillaTableroPicto(text=str(palabra), source=f'./tableros/pictogramas/{imagen}', on_press=self.on_button_press)
                else:
                    btn = CasillaTablero(text=str(palabra), on_press=self.on_button_press, font_name='Texto')
                self.casillas.append(btn)
                self.add_widget(btn)

    def on_button_press(self, instance):
        if self.pictos:
            self.controlador.on_casilla_tab(instance.label.text)
        else:
            self.controlador.on_casilla_tab(instance.text)