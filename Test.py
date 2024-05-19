from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from Custom import ButtonRnd
from kivy.uix.label import Label
from kivy.graphics import InstructionGroup
import numpy as np
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter

class Pelota(Scatter):
    def __init__(self, esquina, controlador, size_sc, label, **kwargs):
        super(Pelota, self).__init__(**kwargs)
        self.esquina = esquina
        self.controlador = controlador
        self.size_sc = size_sc
        self.bind(pos=self.update_limites)
        self.label = label

    def update_limites(self, *args):
        valor_x = self.center_x / self.size_sc[0]
        valor_y = self.center_y / self.size_sc[1]
        self.controlador.set_limite(valor_x, self.esquina, 0)
        self.controlador.set_limite(valor_y, self.esquina, 1)
        self.label.text = f'{self.esquina}: ({valor_x:.2f}, {valor_y:.2f})'



class Test(Screen):
    def __init__(self, controlador, **kwargs):
        super(Test, self).__init__(**kwargs)
        self.controlador = controlador
        self.background_color = (0, 0, 0, 1) 
        self.controlador.set_escanear(False)

        texto_central = ("Mire al rededor de la pantalla y parpadee para confirmar el calibrado\n"+
                        "Es un modelo en desarrollo, cuantos mas datos recopilemos, mejor funcionara en un futuro\n"+
                        "El punto rojo deberia seguir su mirada, al parpadear debería escuchar un sonido de click y ver el punto verde")

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)

        self.layout = BoxLayout(orientation='vertical')

        # El boton de inicio
        btn_inicio = ButtonRnd(text='Inicio', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1}, on_press= self.on_inicio, font_name='Texto')
        self.layout.add_widget(btn_inicio)

        # El texto explicativo
        self.texto_explicativo = Label(text=texto_central, font_size=self.controlador.get_font_txts(),halign='center', size_hint=(1, .8), font_name='Texto')
        self.layout.add_widget(self.texto_explicativo)

        # Variables reconocidas
        self.texto_variables = Label(text="", font_size=self.controlador.get_font_txts(), size_hint=(1, .1), font_name='Texto')
        self.layout.add_widget(self.texto_variables)

        # The image box
        self.image_box = Image(size_hint=(.2, .3), pos_hint={'right': 1, 'top': 0.1})
        self.layout.add_widget(self.image_box)

        self.add_widget(self.layout)

        self.primera = True



    # Funcion para dibujar el circulo amarillo una vez abierta la ventana(para centrarlo bien)
    def on_enter(self, *args):  
        # Se crea el circulo rojo y se añade
        self.circulo = Ellipse(pos=self.center, size=(50, 50))
        with self.canvas:
            self.circulo_color = Color(1, 0, 0)  # Rojo
            self.circulo_instr = InstructionGroup()
            self.circulo_instr.add(self.circulo_color)
            self.circulo_instr.add(self.circulo)
            if self.circulo_instr not in self.layout.canvas.children:
                self.layout.canvas.add(self.circulo_instr)
        self.controlador.set_escanear(True)

        if self.primera:
            self.primera = False
            limites = self.controlador.get_limites()
            esquinas = ['Esquina inferior izquierda', 'Esquina inferior derecha', 'Esquina superior izquierda', 'Esquina superior derecha', 'Centro']
            # Añadir las pelotas
            for i in range(5):
                # Crear una pelota
                label = Label(text=f'{esquinas[i]}: {limites[i]}', size_hint=(.2, .1), pos_hint={'x': 0, 'top': 1 - i * 0.2})
                pelota = Pelota(i, self.controlador, self.size, label, size=(20, 20), size_hint=(None, None))
                with pelota.canvas:
                    Color(0, 1, 0)  # Verde
                    Ellipse(size=pelota.size)
                pelota.center = (limites[i][0] * self.width, limites[i][1] * self.height)
                self.add_widget(pelota)

                # Añadir el texto con los valores de las esquinas
                self.layout.add_widget(label)

                # Vincular la pelota con su etiqueta
                pelota.label = label

        # Añade las tareas de actualización al reloj
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  

    # Funcion para dejar de escanear y volver al inicio
    def on_inicio(self, *args):
        self.controlador.change_screen('inicio')

    def on_leave(self, *args):
        self.controlador.set_escanear(False)
        self.circulo_instr.clear()
        Clock.unschedule(self.update_image_box)
        Clock.unschedule(self.update)

    # Funcion para actualizar la posición del círculo y el color
    def update(self, dt):
        # Si se ha activado el escaneo y estamso en esta pantalla
        if self.controlador.get_escanear() and self.controlador.get_screen() == 'test':
            # Obtiene la posición de la mirada y el ear
            datos = self.controlador.obtener_posicion_mirada_ear()

            # Si no se detecta cara, no hacer nada
            if datos is None:
                self.controlador.mensaje("No se detecta cara")
                return
            
            # Desempaqueta los datos
            proxima_pos_t, click = datos
            # Actualiza el color del círculo
            if click == 0:  # Rojo
                self.circulo_color.rgba = (1, 0, 0, 1)
            elif click == 1:  # Verde
                self.circulo_color.rgba = (0, 1, 0, 1)

            # La posición del circulo no excede las dimensiones de la pantalla
            proxima_pos_t = np.minimum(proxima_pos_t*self.size, self.size - np.array([50, 50]))
            
            #Indice 2 ya que tiene el color en 1, no como recopilar:
            self.circulo_instr.children[2].pos =  (proxima_pos_t).flatten()


    def update_image_box(self, dt):
            # Only update the image box in calibration state 0
            frame = self.controlador.get_frame_editado(0.15)

            # Convert the frame to a texture
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(frame.tostring(), colorfmt='bgr', bufferfmt='ubyte')

            # Invertir la imagen verticalmente
            texture.flip_vertical()
            self.image_box.texture = texture

