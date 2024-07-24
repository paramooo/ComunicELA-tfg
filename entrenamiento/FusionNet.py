import torch
import torch.nn as nn


class FusionNet(nn.Module):
    def __init__(self, ann, cnn):
        super(FusionNet, self).__init__()
        self.ann = ann
        self.cnn = cnn
        self.fusion_layer = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Sigmoid()
        )

    def forward(self, x_ann, x_cnn):
        out_ann = self.ann(x_ann)
        out_cnn = self.cnn(x_cnn)
        # Concatena las salidas de las dos redes
        fusion = torch.cat((out_ann, out_cnn), dim=1)
        # Pasa la concatenación a través de la capa de fusión
        out = self.fusion_layer(fusion)
        return out

