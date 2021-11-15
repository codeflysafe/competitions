import logging
import os
import numpy as np
from numpy.lib.npyio import save
import torch
import random
from torch._C import dtype
import yaml
from logger import init_log
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_config(config_path:str,save_log = False):
    # 获取当前脚本所在文件夹路径
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yaml_path = os.path.join(curPath, config_path)
    # open方法打开直接读出来
    config = yaml.load(open(yaml_path, 'r', encoding='utf-8'),Loader=yaml.SafeLoader)
    logger = init_log(config['model']['name'],save_log = save_log)
    return config,logger


def generator_char_dict(characters):
    """
    生成对应的词典
    """
    CHAR2LABEL = {char: i for i, char in enumerate(characters)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    num_class = len(LABEL2CHAR)
    return CHAR2LABEL,LABEL2CHAR,num_class


def init(seed:int,config_path:str,save_log = True):
    setup_seed(seed)
    config, logger =  parse_config(config_path,save_log = save_log)
    CHAR2LABEL,LABEL2CHAR,num_class = generator_char_dict(config['base']['characters'])
    config['base']['char2labels'] = CHAR2LABEL
    config['base']['labels2char'] = LABEL2CHAR
    config['model']['num_class'] = num_class
    logger.info(f'Config: {config}')
    #print(type(config['base']['char2labels']))
    return config,logger


def save_preds(name,reals, preds,images_path,output_path):
    res = pd.DataFrame({'Image_path':images_path,'Real':reals,'Pred':preds})
    curPath = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(curPath,f'predicts/{name}_{output_path}.csv')
    res.to_csv(save_path,index=False)
    return save_path


if __name__ == '__main__':
    parse_config('configs/local/resnet_rnn_ctc.yaml')