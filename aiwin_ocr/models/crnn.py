import torch.nn as nn
from models import nets
from models.base_model import BaseModule

class CRNN(BaseModule):

    def __init__(self, config: dict):
        super().__init__(config)
       
        self.cnn = getattr(nets,self.config['model']['cnn'])(self.config['base']['in_channel'])
        if self.config['model']['rnn'] == 'lstm':
            self.rnn = nn.LSTM(self.config['model']['map_to_seq_hidden'], self.config['model']['rnn_hidden'], 
             num_layers=self.config['model']['rnn_num_layers'], bidirectional=True)
        else:
            self.rnn = nn.GRU(self.config['model']['map_to_seq_hidden'], self.config['model']['rnn_hidden'], 
             num_layers=self.config['model']['rnn_num_layers'], bidirectional=True)
        self.dense = nn.Linear(self.config['model']['rnn_num_layers']*self.config['model']['rnn_hidden'],
              self.config['model']['num_class'])
        
    
    def forward(self,images):
        """
        images: 训练数据, shape (batch_size, channel, height, width)
        height 必须为 32
        """
        b,c,h,w = images.shape
        assert h == 32
        x = self.cnn(images)
        b,c,h,w = x.shape
        x = x.view(b, c* h, w)
        x = x.permute(2, 0, 1)  # (width, batch, feature)
        x, _ = self.rnn(x)
        output = self.dense(x)
        return output  # shape: (seq_len, batch, num_class)

    