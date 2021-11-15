# training model
import logging
from util import init
import torch
import torch.optim as optim
from dataset import train_loader
from models import Trainer
import torch.nn as nn
from tqdm import tqdm

def main(config:dict,logger:logging):
    # 更新配置文件
    config['base']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    train_load,valid_load = train_loader(config,logger)

    # 初始化模型以及参数等
    trainer = Trainer(config,logger,config['base']['labels2char'])
    optimizer = optim.Adam(trainer.parameters(), lr=config['optimizer']['base_lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['optimizer']['milestones'],
               gamma = config['optimizer']['gamma'])
    criterion = nn.CTCLoss(reduction='sum')
    criterion.to(config['base']['device'])

 
    # 训练模型
    for epoch in tqdm(range(1, config['train']['epochs']+1)):
        if trainer.runing():
            trainer.train(optimizer,scheduler,criterion,epoch,train_load)
            if epoch % config['train']['valid_interval'] == 0:
                trainer.val(criterion,epoch,valid_load)
                trainer.info()
        else:
            break
    
    logger.info(f'All Over!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train captcha model')
    parser.add_argument('--mode', '--m', type=str,required= False,default= 'local', help='local or remote')
    parser.add_argument('--config_path','--c',type=str,required= False,default= 'resnet_rnn_ctc.yaml',help="config path for training")
    args = parser.parse_args()
    config, logger = init(100, f'configs/{args.mode}/{args.config_path}',args.mode == 'remote')
    assert config['base']['train'] == True
    main(config,logger)