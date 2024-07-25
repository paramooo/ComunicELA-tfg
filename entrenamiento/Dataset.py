from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Conjuntos import Conjuntos
from PIL import Image
from torchvision.transforms import ToTensor

class Dataset(Dataset):
    def __init__(self, img_dir, txt_input_file, txt_output_file, sigma, transform=None, conjunto=None):
        self.img_dir = img_dir
        # Cargar los datos de texto que no ocupan tanto espacio en el init
        self.txt_input_data = np.loadtxt(txt_input_file, delimiter=',')
        self.txt_output_data = np.loadtxt(txt_output_file, delimiter=',')
        self.file_names = os.listdir(img_dir)
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
        return len(self.file_names)

    def __getitem__(self, idx):
        # Las imagenes se cargan aqui para ahorrar memoria
        img_name = os.path.join(self.img_dir, self.file_names[idx])
        image = Image.open(img_name).convert('L').convert('F')
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=-1) 

        # Cogemos los inputs y outputs
        txt_input_data = self.txt_input_data[idx]
        txt_output_data = self.txt_output_data[idx]

        # Convertir la imagen a un tensor de tipo float
        image = ToTensor()(image).float()

        return txt_input_data, image, txt_output_data