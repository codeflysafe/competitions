import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleCNN(nn.Module):
    """
    Simple CRNN
    """
    def __init__(self,in_channel):
        super(SimpleCNN, self).__init__()
        self.in_channel = in_channel
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()
        def _cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = self.in_channel
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                _cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)
        self.layers = nn.Sequential(modules)
    

    def forward(self, x):
        # shape of images: (batch, channel, height, width)
        return self.layers(x)

if __name__ == '__main__':
    x = torch.rand((16,3,32,100))
    net = SimpleCNN(in_channel=3)
    out = net(x)
    # 16,256,1,6
    print(out.shape,x.shape)