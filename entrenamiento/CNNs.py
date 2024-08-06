import torch.nn as nn

class CNNs:
    def __init__(self):
        pass

    def dividir(self, ancho, alto):
        return int(ancho/2), int(alto/2)


    # -------------------------------------------------- PRIMERAS OBSERVACIONES --------------------------------------------------
    def crear_cnn_1(self):
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(4*ancho*alto, 80))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(80, 2))
      model.add_module('output', nn.Sigmoid())
      return model


    def crear_cnn_2(self):
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(8*ancho*alto, 100))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(100, 2))
      model.add_module('output', nn.Sigmoid())
      return model


    def crear_cnn_3(self):
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(8*ancho*alto, 150))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(150, 100))
      model.add_module('relu4', nn.ReLU())
      model.add_module('fc3', nn.Linear(100, 2))
      model.add_module('output', nn.Sigmoid())
      return model



    def crear_cnn_4(self):
        ancho, alto = (210,120)
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

    def crear_cnn_5(self, tamaño):
        ancho, alto = tamaño
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
    


# -------------------------------------------------- SEGUNDAS OBSERVACIONES --------------------------------------------------
    def crear_cnn_2_1(self):
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(4*ancho*alto, 80))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(80, 2))
      return model
    
    def crear_cnn_2_2(self):
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1))
      model.add_module('bn1', nn.BatchNorm2d(8))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(8*ancho*alto, 100))
      model.add_module('dropout1', nn.Dropout(p=0.5))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(100, 2))
      return model

    def crear_cnn_2_3(self):  
      ancho, alto = (210,120)
      model = nn.Sequential()
      model.add_module('conv1', nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1))
      model.add_module('relu1', nn.ReLU())
      model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
      ancho, alto = self.dividir(ancho, alto)
      model.add_module('flatten', nn.Flatten())
      model.add_module('fc1', nn.Linear(8*ancho*alto, 150))
      model.add_module('relu3', nn.ReLU())
      model.add_module('fc2', nn.Linear(150, 100))
      model.add_module('relu4', nn.ReLU())
      model.add_module('fc3', nn.Linear(100, 2))
      return model
    
    def crear_cnn_2_4(self):
        ancho, alto = (210,120)
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
        model.add_module('fc1', nn.Linear(16*ancho*alto, 2))
        return model

    def crear_cnn_2_5(self):
        ancho, alto = (210, 120)
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('bn1', nn.BatchNorm2d(16))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        model.add_module('bn2', nn.BatchNorm2d(32))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ancho, alto = self.dividir(ancho, alto)
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(32 * ancho * alto, 200))
        model.add_module('dropout1', nn.Dropout(p=0.5))
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(200, 300))
        model.add_module('dropout2', nn.Dropout(p=0.5))
        model.add_module('relu4', nn.ReLU())
        model.add_module('fc3', nn.Linear(300, 100))
        model.add_module('dropout3', nn.Dropout(p=0.5))
        model.add_module('relu5', nn.ReLU())
        model.add_module('fc4', nn.Linear(100, 2))
        return model

