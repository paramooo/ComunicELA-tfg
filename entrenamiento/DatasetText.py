from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Conjuntos import Conjuntos
from PIL import Image
from torchvision.transforms import ToTensor

class DatasetText(Dataset):
    def __init__(self, txt_input_file, txt_output_file, sigma, transform=None, conjunto=None):
        # Cargar los datos de texto que no ocupan tanto espacio en el init
        self.txt_input_data = np.loadtxt(txt_input_file, delimiter=',')
        self.txt_output_data = np.loadtxt(txt_output_file, delimiter=',')
        self.sigma = sigma
        self.transform = transform
        self.conjunto = conjunto

        # Suavizar los datos de entrada de texto 
        for i in range(self.txt_input_data.shape[1]):
            self.txt_input_data[:, i] = gaussian_filter1d(self.txt_input_data[:, i], self.sigma)

        # Transformar los datos al conjunto de entrenamiento
        if conjunto is not None:
            normalizar_funcion = getattr(Conjuntos, f'conjunto_{self.conjunto}')
            self.txt_input_data = normalizar_funcion(self.txt_input_data)
        
        self.transform = transform or ToTensor()

    def __len__(self):
        return len(self.txt_input_data)

    def __getitem__(self, idx):
        # Cogemos los inputs y outputs
        txt_input_data = self.txt_input_data[idx].astype(np.float32)
        txt_output_data = self.txt_output_data[idx].astype(np.float32)


        return txt_input_data, txt_output_data