# training model
import logging
from util import init,save_preds
import os
import torch
import torch.optim as optim
from dataset import test_loader, train_loader
from models import Trainer
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

def main(config:dict,logger:logging):
    # 更新配置文件
    config['base']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练数据
    test_load = test_loader(config,logger)

    # 初始化模型以及参数等
    trainer = Trainer(config,logger,config['base']['labels2char'],eval = True)
    criterion = nn.CTCLoss()
    criterion.to(config['base']['device'])
    
    # 预测数据
    reals, preds = trainer.predict(test_load) 
    images_path = test_load.dataset.image_paths     
    test_pre = config['test']['input_path'].split('/')[-1]
    
    # 存储结果
    save_path = save_preds(test_pre,reals,preds,images_path,config['test']['output_path'])
    logger.info(f'save predict result in {save_path}!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train captcha model')
    parser.add_argument('--mode', '--m', type=str,required= False,default= 'local', help='The mode of train')
    parser.add_argument('--config_path','--c',type=str,required= False,default= 'resnet_rnn_ctc.yaml',help="config path for training")
    args = parser.parse_args()
    config, logger = init(100, f'configs/{args.mode}/{args.config_path}')
    config['base']['train'] = True
    assert config['base']['pretrained'] == True and config['base']['load_path']
    main(config,logger)