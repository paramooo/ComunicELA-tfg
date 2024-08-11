from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.switch import Switch
from KivyCustom.Custom import ButtonRnd, CustomSpinner
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.modalview import ModalView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox
from KivyCustom.PopUp import CustPopup
from KivyCustom.PopUpAjustes import PopUpAjustes


class Inicio(Screen):
    def __init__(self, controlador, **kwargs):
        super(Inicio, self).__init__(**kwargs)
        self.controlador = controlador
        self.primera = True

        # Crea una imagen de fondo
        self.fondo = Image(source=self.controlador.get_fondo() , allow_stretch=True, keep_ratio=False)
        self.add_widget(self.fondo)
        
        # Alineamos horizontalmente para separar en dos
        Principal = BoxLayout(orientation='vertical', padding=20)

        TituloIdioma = BoxLayout(orientation='horizontal', size_hint=(1, 0.2))

        titulo = Label(text='ComunicELA', font_size='100sp', halign='center', color=(1, 1, 1, 1), size_hint=(0.85, 1), font_name='Titulo')

        idioma = BoxLayout(orientation='vertical', size_hint=(0.05, 1))

        self.imagenIdioma = Button(
            background_normal=self.controlador.get_idioma_imagen(),
            background_down=self.controlador.get_idioma_imagen(),  # Mantén la imagen al hacer clic
            background_color=(1, 1, 1, 1),  # Color de fondo transparente
            size_hint=(1, 0.8)
        )        
        self.imagenIdioma.bind(on_release=self.on_idioma_click)

        self.textIdioma = Label(text=self.controlador.get_idioma_string(), font_size='22sp', halign='center', valign='middle', color=(1, 1, 1, 1), size_hint=(1, 0.5), font_name='Texto')

        caja = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))

        # Parte izquierda con los botones y el titulo
        self.Izquierda = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=20)



        #Botones
        # Menu de seleccion de camara
        self.camera_spinner = CustomSpinner(
            text=self.controlador.get_string('cargando_camara'),
            values=[],
            #size_hint=(0.6, 0.1),
            pos_hint={'center_x': 0.5},
            font_name='Texto', 
            font_size=20)
        
        # Menu de seleccion de camara
        self.voz_spinner = CustomSpinner(
            text=self.controlador.get_string('cargando_voces'),
            values=[],
            #size_hint=(0.6, 0.1),
            pos_hint={'center_x': 0.5},
            font_name='Texto', 
            font_size=20)
        
        estado = self.controlador.get_estado_corrector(mensajes = False)
        if estado == None:
            textoboton = (self.controlador.get_string('nodisp') if estado else self.controlador.get_string('nodisp'))
        else:
            textoboton = (self.controlador.get_string('activado') if estado else self.controlador.get_string('desactivado'))
        self.boton_gemini = ButtonRnd(text=textoboton, size_hint=(0.6, 1), on_press=lambda x: self.on_corrector(), font_name='Texto', font_size=self.controlador.get_font_txts(), pos_hint={'center_x': 0.5})
        
        self.btn_cal = ButtonRnd(text=self.controlador.get_string('btn_inicio_calibrar'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_cam('calibrar'), font_name='Texto', font_size='35sp')

        self.btn_tst = ButtonRnd(text=self.controlador.get_string('btn_inicioDes_test'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('test'), font_name='Texto', font_size='35sp')

        self.btn_rec = ButtonRnd(text=self.controlador.get_string('btn_inicioDes_reco'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_r('recopilar') , font_name='Texto', font_size='35sp')

        self.btn_ree = ButtonRnd(text=self.controlador.get_string('btn_inicio_reentrenar'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_r('reentrenarinstruc'), font_name='Texto', font_size='35sp')

        self.btn_tab = ButtonRnd(text=self.controlador.get_string('btn_inicio_tableros'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen('tabinstruc'), font_name='Texto', font_size='35sp')

        self.btn_pruebas = ButtonRnd(text=self.controlador.get_string('btn_inicioDes_pruebas'), size_hint=(1, 0.2), on_press=lambda x: self.controlador.change_screen_r('tablerosprueb'), font_name='Texto', font_size='35sp')

        self.txt_des = Label(text=self.controlador.get_string('mensaje_inicio_op_des'), halign='center', size_hint=(1, 0.1), font_size='20sp')
        
        self.camera_spinner.bind(text=self.seleccionar_camara)
        self.voz_spinner.bind(text=self.gestionar_spinner_voces)

        espacio_blanco2 = BoxLayout(size_hint=(1, 0.05))

        # Parte derecha con el texto y la imagen
        Derecha = BoxLayout(orientation='vertical', size_hint=(0.5, 1), spacing=10)
       
        self.image_box = Image(size_hint=(0.7, 0.7), pos_hint={'center_x': 0.5}, allow_stretch=True, keep_ratio=True)

        self.popajustes = PopUpAjustes(self.camera_spinner, self.voz_spinner, self.boton_gemini, self.controlador)
        self.boton_popajustes = ButtonRnd(text=self.controlador.get_string('ajustes'), size_hint=(0.6, 0.1), on_press=lambda x: self.popajustes.open(), font_name='Texto', font_size=self.controlador.get_font_txts(), pos_hint={'center_x': 0.5})

        # Montamos la estructura
        self.Izquierda.add_widget(self.btn_cal)
        self.Izquierda.add_widget(self.btn_ree)
        self.Izquierda.add_widget(self.btn_tab)
        self.Izquierda.add_widget(espacio_blanco2)

        Derecha.add_widget(Widget(size_hint_y=0.01))    
        Derecha.add_widget(self.image_box)

        Derecha.add_widget(Widget(size_hint_y=0.04))  
        Derecha.add_widget(self.boton_popajustes)  
        Derecha.add_widget(Widget(size_hint_y=0.12))

        
        caja.add_widget(self.Izquierda)
        caja.add_widget(Derecha)
        idioma.add_widget(Widget(size_hint_y=0.1))
        idioma.add_widget(self.imagenIdioma)
        idioma.add_widget(self.textIdioma)
        idioma.add_widget(Widget(size_hint_y=0.6))
        TituloIdioma.add_widget(Widget(size_hint_x=0.1))
        TituloIdioma.add_widget(titulo)
        TituloIdioma.add_widget(idioma)
        TituloIdioma.add_widget(Widget(size_hint_x=0.02))

        Principal.add_widget(TituloIdioma)
        
        Principal.add_widget(caja)

        self.add_widget(Principal)

        self.tutorial_buttons = [
            (self.camera_spinner, self.controlador.get_string('mensaje_tutorial_1')),
            (self.btn_cal, self.controlador.get_string('mensaje_tutorial_2')),
            (self.btn_tab, self.controlador.get_string('mensaje_tutorial_3')),
            (self.btn_ree, self.controlador.get_string('mensaje_tutorial_4')),
        ]


    def show_tutorial(self, *args):
        if self.tutorial_buttons:
            button, message = self.tutorial_buttons.pop(0)
            
            # Calcula la posición normalizada
            if button == self.camera_spinner:
                pos = 0.5, 0.22
            else:
                pos = (button.center_x / Window.width) + 0.4, button.center_y / Window.height
            
            # Muestra el popup con el mensaje
            show_switch = len(self.tutorial_buttons) == 0
            CustPopup(message, self.show_tutorial, pos, self.controlador, show_switch=show_switch).open()

    def on_corrector(self):
        estado = self.controlador.cambiar_estado_corrector()
        if estado is None:        
            textoboton =(self.controlador.get_string('nodisp'))
        else:
            textoboton = (self.controlador.get_string('activado') if estado else self.controlador.get_string('desactivado'))
        self.boton_gemini.text = textoboton



    def _keyboard_closed(self):
        if self._keyboard is not None:
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'd':
            if self.controlador.get_desarrollador() == False:
                self.Izquierda.add_widget(self.btn_tst) 
                self.Izquierda.add_widget(self.btn_rec) 
                self.Izquierda.add_widget(self.btn_pruebas)
                self.Izquierda.add_widget(self.txt_des)
                self.controlador.set_desarrollador(True)
            else:
                self.Izquierda.remove_widget(self.txt_des)
                self.Izquierda.remove_widget(self.btn_tst)  
                self.Izquierda.remove_widget(self.btn_rec) 
                self.Izquierda.remove_widget(self.btn_pruebas)
                self.controlador.set_desarrollador(False)
        #Escape para salir
        if keycode[1] == 'escape':
            self.controlador.salir()
        return True
    
    def on_enter(self, *args):
        # Menu de seleccion de camara una vez dentro para asi poder actualizar las camaras
        if self.primera:
            self.controlador.obtener_camaras(stop = False)
            self.get_voces()          
            self.primera = False
    
        # Schedule the update of the image box every 1/30 seconds
        Clock.schedule_interval(self.update_image_box, 1.0 / 30)

        # Llamar al método show_tutorial después de que la vista inicial se haya completado
        if self.controlador.get_show_tutorial():
            Clock.schedule_once(self.show_tutorial, 1)
        
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)


    def seleccionar_camara(self, _, text, primera = False):
        if text.startswith(self.controlador.get_string('camara')):
            camara = text.split(' ')[1]
            if camara == self.controlador.get_string('principal'):
                camara = 0
            #AQUI AL ACTUALIZAR EL NOMBRE A LA CAMARA DEL CONFIG SE VUELVE A PONER AUNQ YA ESTA ACTIVA
            self.controlador.seleccionar_camara(int(camara))
        elif text == self.controlador.get_string('actualizar_camaras'):
            self.controlador.seleccionar_camara(None)
            self.controlador.obtener_camaras()
        elif text == self.controlador.get_string('btn_inicioDes_seleccionarCam') or text == self.controlador.get_string('cargando_camara'):
            pass

    def gestionar_spinner_voces(self, spinner, text):
        if text == self.controlador.get_string('actualizar_voces'):
            self.get_voces()
        elif text != self.controlador.get_string('btn_inicioDes_seleccionarVoz'):
            self.seleccionar_voz(spinner, text)
    
    def get_voces(self):
        self.voz_spinner.text = self.controlador.get_string('cargando_voces')
        self.voz_spinner.values = self.controlador.get_voces()
        voz = self.controlador.get_voz_seleccionada()
        if voz is not None:
                self.voz_spinner.text = voz
        else:
             self.voz_spinner.text = self.controlador.get_string('btn_inicioDes_seleccionarVoz')


    def seleccionar_voz(self, _, text):
        self.controlador.seleccionar_voz(text)
        self.voz_spinner.text = text
        
        
    def update_image_box(self, dt):
        frame = self.controlador.get_frame_editado()
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
        self._keyboard_closed()


    def on_idioma_click(self, *args):
        self.controlador.cambiar_idioma()
        imagen = self.controlador.get_idioma_imagen()
        self.imagenIdioma.background_normal = imagen
        self.textIdioma.text = self.controlador.get_idioma_string()
        self.btn_cal.text = self.controlador.get_string('btn_inicio_calibrar')
        self.btn_ree.text = self.controlador.get_string('btn_inicio_reentrenar')
        self.btn_tab.text = self.controlador.get_string('btn_inicio_tableros')
        self.btn_pruebas.text = self.controlador.get_string('btn_inicioDes_pruebas')
        self.txt_des.text = self.controlador.get_string('mensaje_inicio_op_des')
        self.btn_rec.text = self.controlador.get_string('btn_inicioDes_reco')
        self.btn_tst.text = self.controlador.get_string('btn_inicioDes_test')
        estado = self.controlador.get_estado_corrector()
        if estado == None:
            textoboton =(self.controlador.get_string('nodisp'))
        else:
            textoboton = (self.controlador.get_string('activado') if estado else self.controlador.get_string('desactivado'))
        
        # Cambiar el texto del popup de ajustes
        self.boton_gemini.text = textoboton
        self.boton_popajustes.text = self.controlador.get_string('ajustes')
        self.popajustes.close_button.text = self.controlador.get_string('cerrar')
        self.popajustes.label_select_camera.text = self.controlador.get_string('seleccion_camara')
        self.popajustes.label_select_voz.text = self.controlador.get_string('seleccion_voz')
        self.popajustes.label_gemini.text = self.controlador.get_string('conjugar')
        self.popajustes.label_titulo.text = self.controlador.get_string('ajustes')