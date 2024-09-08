import os
import cv2
from tqdm import tqdm
import sys


#Importamos el detector
sys_copy = sys.path.copy()
sys.path.append('./')
from Servicios.Detector import Detector
sys.path = sys_copy


class EditorFrames:
    """
    Clase encargada de recortar los frames a la zona de interés y redimensionarlos a un tamaño fijo

    """


    def __init__(self, ratio, ancho, altoArriba, altoAbajo):
        """
        Inicializa el editor de frames

        Args:
            ratio (tuple): Relación de aspecto de los frames
            ancho (int): Ancho que se añade horizontalmente desde las pupilas
            altoArriba (int): Lo que se añade verticalmente hacia arriba desde las pupilas
            altoAbajo (int): Lo que se añaade verticalmente hacia abajo desde las pupilas
        
        """
        self.ratio_ancho, self.ratio_alto = ratio
        self.ratio = self.ratio_ancho / self.ratio_alto
        self.ancho = ancho
        self.altoArriba = altoArriba
        self.altoAbajo = altoAbajo
        self.detector = Detector()



    def normalizar_frame(self, frame, coord_o_izq, coord_o_der):
        """
        Función que recorta el frame a la zona de los ojos y redimensiona a un tamaño fijo

        Args:
            frame (np.array): Frame a editar
            coord_o_izq (tuple): Coordenadas del ojo izquierdo
            coord_o_der (tuple): Coordenadas del ojo derecho

        Returns:
            np.array: Frame editado

        """
        # Coordenadas de los ojos
        x_o_izq, y_o_izq = coord_o_izq[0]
        x_o_der, y_o_der = coord_o_der[0]

        # Coordenadas del rectangulo
        x1 = min(x_o_izq, x_o_der)-self.ancho
        x2 = max(x_o_izq, x_o_der)+self.ancho
        y1 = min(y_o_izq, y_o_der)-self.altoArriba
        y2 = max(y_o_izq, y_o_der)+self.altoAbajo

        # Calcular la relación de aspecto actual
        ratio_act = (x2 - x1) / (y2 - y1)

        # Calcular cuántos píxeles se deben agregar a cada lado
        if ratio_act < self.ratio:
            diff = int(((y2 - y1) * self.ratio - (x2 - x1)) / 2)
            x1 -= diff
            x2 += diff
        elif ratio_act > self.ratio:
            diff = int(((x2 - x1) / self.ratio - (y2 - y1)) / 2)
            y1 -= diff
            y2 += diff

        # Recortar el rectangulo de los ojos y redimensionar a un tamaño fijo
        rect_frame = frame[y1:y2, x1:x2]
        rect_frame = cv2.resize(rect_frame, (self.ratio_ancho, self.ratio_alto), interpolation = cv2.INTER_AREA)

        return rect_frame



    def editar_frames(self):
        """
        Función que edita todos los frames de una carpeta y los guarda en otra
        
        """
        #Crear la carpeta si no existe
        if not os.path.exists(f'./entrenamiento/datos/frames/recortados/{self.ancho}-{self.altoArriba}-{self.altoAbajo}'):
            os.makedirs(f'./entrenamiento/datos/frames/recortados/{self.ancho}-{self.altoArriba}-{self.altoAbajo}')
        
        archivos = os.listdir('./entrenamiento/datos/frames/total')
        for nombre_archivo in tqdm(archivos, desc="Procesando frames"):
            frame = cv2.imread(os.path.join('./entrenamiento/datos/frames/total', nombre_archivo))
            datos = self.detector.obtener_coordenadas_indices(frame)
            frame_editado = self.normalizar_frame(frame, datos[0], datos[1])
            cv2.imwrite(os.path.join(f'./entrenamiento/datos/frames/recortados/{self.ancho}-{self.altoArriba}-{self.altoAbajo}', nombre_archivo), frame_editado)
