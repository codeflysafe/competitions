import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModule

class FineResnetRnn(BaseModule):
    """
    微调 resnet_rnn
    """
    def __init__(self,pretrained_model:nn.Module,fixed_layer = 2):
        super(FineResnetRnn, self).__init__()
        self.fixed_layer = fixed_layer
        self.pretrained_model = pretrained_model
        self._fixed_layer()

    def _fixed_layer(self):
        for param in self.pretrained_model.cnn.conv1.parameters():
            param.requires_grad = False
        for idx, child in enumerate(self.pretrained_model.cnn.layers.children()):
            if idx < self.fixed_layer:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, images):
        # shape of images: (batch, channel, height, width)
        return self.pretrained_model(images)

    def name(self):
        return "fine_resnet_rnn"



