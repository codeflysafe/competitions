import torch.nn as nn
import torch 
import time
import os

class BaseModule(nn.Module):
    """
    Base Module for network
    """
    def __init__(self,config:dict):
        super().__init__()
        self.config = config

    def name(self):
        return self.config['model']['name']
    
    def load(self):
        """
        加载模型
        """
        if self.config['base']['pretrained'] or not self.config['base']['train']:
            self.load_state_dict(torch.load(self.config['base']['load_path'], map_location=self.config['base']['device']))
            return True
        return False
    
    def save(self):
        """

        """
        assert self.config['train']['checkpoints_path']
        pid = os.getpid()
        day = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_path = os.path.join(self.config['train']['checkpoints_path'],f'{day}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_model_path = os.path.join(model_path,f'{day}_{pid}_{self.name()}.pt')
        torch.save(self.state_dict(), save_model_path)
        return save_model_path