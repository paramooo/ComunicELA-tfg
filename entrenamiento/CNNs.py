import torch.nn as nn

class CNNs:
    def __init__(self):
        pass

    def dividir(self, ancho, alto):
        return int(ancho/2), int(alto/2)
    # ------------------------------ Imagenes de 200x50 ------------------------------

    # - Capa convolucional 1 -> 16 filtros de 3x3
    # - Capa de pooling 1 -> MaxPool de 2x2
    # - Capa totalmente conectada
    def crear_cnn_1(self, tamaño_img):
        ancho, alto = tamaño_img
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(16*ancho*alto, 150))
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(150, 2))
        model.add_module('output', nn.Sigmoid())
        return model
    
    # - Capa convolucional 1 -> 16 filtros de 3x3
    # - Capa de Batch Normalization
    # - Capa de activación ReLU
    # - Capa de pooling 1 -> MaxPool de 2x2
    # - Capa totalmente conectada
    # - Capa de activación ReLU
    # - Capa de Dropout
    # - Capa totalmente conectada
    # - Capa de activación Sigmoid

    def crear_cnn_2(self, tamaño_img):
        ancho, alto = tamaño_img
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('bn1', nn.BatchNorm2d(16))  # Batch Normalization
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(16*ancho*alto, 150))
        model.add_module('relu3', nn.ReLU())
        model.add_module('dropout1', nn.Dropout(0.25))  # Dropout
        model.add_module('fc2', nn.Linear(150, 2))
        model.add_module('output', nn.Sigmoid())
        return model

    # - Capa convolucional 1 -> 16 filtros de 3x3
    # - Capa de pooling 1 -> MaxPool de 2x2
    # - Capa convolucional 16 -> 32 filtros de 3x3
    # - Capa de pooling 2 -> MaxPool de 2x2
    # - Capa totalmente conectada
    def crear_cnn_3(self, tamaño_img):
        ancho, alto = tamaño_img
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(32*ancho*alto, 200))  
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(200, 2))
        model.add_module('output', nn.Sigmoid())
        return model
    

    # - Capa convolucional 1 -> 16 filtros de 3x3
    # - Capa de pooling 1 -> MaxPool de 2x2
    # - Capa convolucional 16 -> 32 filtros de 3x3
    # - Capa de pooling 2 -> MaxPool de 2x2
    # - Capa convolucional 32 -> 64 filtros de 3x3
    # - Capa de pooling 3 -> MaxPool de 2x2
    # - Capa totalmente conectada
    # - Capa totalmente conectada

    def crear_cnn_4(self, tamaño_img):
        ancho, alto = tamaño_img
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('conv2', nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(16*ancho*alto, 250))  #64 filtros, 25x6 tamaño de la imagen despues de 3 maxpool
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(250, 100))
        model.add_module('relu4', nn.ReLU())
        model.add_module('fc3', nn.Linear(100, 2))
        model.add_module('output', nn.Sigmoid())
        return model



