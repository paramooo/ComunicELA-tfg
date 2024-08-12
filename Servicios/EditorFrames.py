import os
import cv2
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
from Servicios.Detector import Detector

class EditorFrames:
    def __init__(self, ratio, ancho, altoArriba, altoAbajo):
        self.ratio_ancho, self.ratio_alto = ratio
        self.ratio = self.ratio_ancho / self.ratio_alto
        self.ancho = ancho
        self.altoArriba = altoArriba
        self.altoAbajo = altoAbajo
        self.detector = Detector()

    def normalizar_frame(self, frame, coord_o_izq, coord_o_der):
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

        # Recortar el rectangulo de los ojos
        rect_frame = frame[y1:y2, x1:x2]

        # Redimensionar a 200x50 manteniendo la relación de aspecto
        rect_frame = cv2.resize(rect_frame, (self.ratio_ancho, self.ratio_alto), interpolation = cv2.INTER_AREA)

        return rect_frame
