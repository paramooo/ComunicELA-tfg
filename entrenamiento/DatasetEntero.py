from torch.utils.data import Dataset
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from Conjuntos import Conjuntos
import torch
import matplotlib.pyplot as plt


class DatasetEntero(Dataset):
    def __init__(self, img_dir, txt_input_file, txt_output_file, sigma, transform=None, conjunto=None, imagenes=False):
        self.img_dir = img_dir
        self.imagenes = imagenes

        # Cargar los datos de texto que no ocupan tanto espacio en el init
        self.txt_input_data = np.loadtxt(txt_input_file, delimiter=',')
        self.txt_output_data = np.loadtxt(txt_output_file, delimiter=',')
        self.personas = np.loadtxt('./entrenamiento/datos/frames/byn/personas.txt')
        self.sigma = sigma
        self.conjunto = conjunto

        # Suavizar los datos de entrada de texto
        for i in range(self.txt_input_data.shape[1]):
            self.txt_input_data[:, i] = gaussian_filter1d(self.txt_input_data[:, i], self.sigma)

        # Borrar los datos con el ojo cerrado
        indices = np.where(self.txt_input_data[-2]<self.txt_input_data[-1])[0]
        self.txt_input_data = np.delete(self.txt_input_data, indices, axis=0)
        self.txt_output_data = np.delete(self.txt_output_data, indices, axis=0)
        self.personas = np.delete(self.personas, indices)

        # Transformar los datos al conjunto de entrenamiento
        if conjunto is not None:
            normalizar_funcion = getattr(Conjuntos, f'conjunto_{self.conjunto}')
            self.txt_input_data = normalizar_funcion(self.txt_input_data)

        # Obtener el .pt de las imagenes y borrar las posiciones de indices tambien
        if self.imagenes:
          img_dir = os.path.join(self.img_dir, 'imagenes.pt')
          self.imgs = torch.load(img_dir)

          self.imgs = torch.stack(self.imgs)
          self.imgs = np.delete(self.imgs, indices, axis=0)

          print("Imagenes cargadas en ram")
          self.imgs = torch.tensor(self.imgs, dtype=torch.float32).to('cuda')

        # Mover los datos a la GPU
        self.txt_input_data = torch.tensor(self.txt_input_data, dtype=torch.float32).to('cuda')
        self.txt_output_data = torch.tensor(self.txt_output_data, dtype=torch.float32).to('cuda')
        print("Datos pasados a gpu")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Cogemos los inputs y outputs
        txt_input_data = self.txt_input_data[idx]
        txt_output_data = self.txt_output_data[idx]
        if self.imagenes:
            return txt_input_data, self.imgs[idx], txt_output_data
        else:
            return txt_input_data, txt_output_data

    def get_indices_persona(self):
      return self.personas

