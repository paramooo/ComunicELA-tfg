import torch.nn as nn

class CNNs:
    def __init__(self):
        pass

    def crear_cnn_1(self):
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        model.add_module('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(32*50*12, 250))  #32 filtros, 50x12 tamaño de la imagen despues de 2 maxpool
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(250, 2))
        model.add_module('output', nn.Sigmoid())
        return model

    def crear_cnn_2(self):
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        model.add_module('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(32*160*120, 250))  # Ajustado para imágenes de 640x480
        model.add_module('relu3', nn.ReLU())
        model.add_module('fc2', nn.Linear(250, 2))
        model.add_module('output', nn.Sigmoid())
        return model
