import torch.nn as nn
from torchvision import models

class CNNs:
  def __init__(self):
      pass

  def dividir(self, ancho, alto, div=2):
      return int(ancho/div), int(alto/div)


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
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
    ancho, alto = self.dividir(ancho, alto)
    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Linear(8*ancho*alto, 100))
    model.add_module('relu3', nn.ReLU())
    model.add_module('fc2', nn.Linear(100, 2))
    return model



  def crear_cnn_3_aux(self, ancho, alto, salidas=2):
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
        model.add_module('fc3', nn.Linear(100, salidas))
        return model

  def crear_cnn_4_aux(self, ancho, alto):
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


  def crear_cnn_2_3(self):
    ancho, alto = (210,120)
    return self.crear_cnn_3_aux(ancho, alto)

  def crear_cnn_2_4(self):
      ancho, alto = (210,120)
      return self.crear_cnn_2_4_aux(ancho, alto)


# ------------------------------------------------------- TERCERAS OBSERVACIONES -----------------------------------


  def crear_cnn_3_3(self):
    ancho, alto = (200,50)
    return self.crear_cnn_3_aux(ancho, alto)

  def crear_cnn_3_4(self):
    ancho, alto = (200,50)
    return self.crear_cnn_4_aux(ancho, alto)

  def crear_cnn_4_3(self):
    ancho, alto = (200,70)
    return self.crear_cnn_3_aux(ancho, alto)

  def crear_cnn_4_4(self):
    ancho, alto = (200,70)
    return self.crear_cnn_4_aux(ancho, alto)


#----------------------------PARA LA FUSIONET-------------------------------------------


  def crear_cnn_f_1(self):
          ancho, alto = (200, 50)
          return self.crear_cnn_3_aux(ancho, alto, salidas=80)

  def crear_cnn_f_2(self):
          ancho, alto = (200, 50)
          return self.crear_cnn_3_aux(ancho, alto, salidas=50)
  
  def crear_cnn_f_3(self):
          ancho, alto = (200, 50)
          return self.crear_cnn_3_aux(ancho, alto, salidas=15)
  


  def crear_resnet(self, indice):
      if indice == 0:
        modelo = models.resnet18(pretrained=True)
      else:
        modelo = models.resnet34(pretrained=True)
      modelo.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
      modelo.fc = nn.Sequential(nn.Linear(modelo.fc.in_features, 2))
      return modelo

  def crear_resnet18(self):
      return self.crear_resnet(0)


  def crear_resnet34(self):
      return self.crear_resnet(1)