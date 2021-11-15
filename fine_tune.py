# 微调模型，数据集为真实数据集
import torch
from models.fine_resnet_rnn import  FineResnetRnn
# training model
import os
import time
from util import  setup_seed, make_model
from  logger import  logger
import torch
import  torch.nn.functional as F
import torch.optim as optim
from torch.nn import CTCLoss
from dataset import train_loader
from models.crnn import CRNN
import config
from config import  configs
from ctc import  ctc_decode
from  tqdm import  tqdm
from logger import  init_log

def train(epoch,show_interval,crnn, optimizer, criterion, device, train_loader):
    """
    :param epoch:
    :param show_interval:
    :param crnn:
    :param optimizer:
    :param criterion:
    :param device:
    :param train_loader:
    :return:
    """
    tot_train_loss = 0.
    tot_train_count = 0
    pbar = tqdm(total=len(train_loader), desc=f"Train, epoch:{epoch}")
    for train_data in train_loader:
        crnn.train()
        images, targets, target_lengths = [d.to(device) for d in train_data]
        logits = crnn(images)
        seq_len, batch, num_class = logits.size()
        log_probs = F.log_softmax(logits, dim=2)
        input_lengths = torch.LongTensor([seq_len] * batch)
        target_lengths = torch.flatten(target_lengths)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        #logger.info(f"loss {loss.item()}, tot_train_count:{tot_train_count}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_train_loss += loss.item()
        tot_train_count += 1
        pbar.update(1)
    pbar.close()
    logger.info(f'Train epoch :{epoch}, train_loss: {tot_train_loss / tot_train_count}')


def valid(epoch,crnn, criterion, device, dataloader,val_acc,early_num,checkpoints_dir,
          decode_method='beam_search', beam_size=10):
    crnn.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    batch_count = 0
    wrong_cases = []
    pbar = tqdm(total=len(dataloader), desc=f"Evaluate,epoch:{epoch}")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, targets, target_lengths = [d.to(device) for d in data]
            logits = crnn(images)
            seq_len, batch, num_class = logits.size()
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.LongTensor([seq_len] * batch)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()
            batch_count += 1
            tot_loss += loss.item()
            target_length_counter = 0
            tot_count += batch
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if real == pred:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))
            pbar.update(1)
        pbar.close()

    valid_loss = tot_loss / batch_count
    valid_acc = tot_correct / tot_count
    logger.info(f'Valid epoch:{epoch}, loss:{valid_loss}, acc:{valid_acc} ')
    if val_acc < valid_acc:
        val_acc = valid_acc
        early_num = 0
        pid = os.getpid()
        day = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_path = os.path.join(checkpoints_dir,f'{day}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_model_path = os.path.join(model_path,
                                       f'fine_tune_{day}_{pid}_{crnn.name()}.pt')
        torch.save(crnn.pretrained_model.state_dict(), save_model_path)
        logger.info(f'save model at {save_model_path}, epoch:{epoch}, loss:{valid_loss},acc:{val_acc} ')
    else:
        early_num += 1
    return val_acc,early_num


def main(train_data_path, model_name,reload_checkpoint):

    assert reload_checkpoint
    crc_train_config = configs[model_name]
    epochs = crc_train_config['epochs']
    m_epochs = crc_train_config['m_epochs']
    lr = crc_train_config['lr']
    m_lr = crc_train_config['m_lr']
    momentum = crc_train_config['momentum']
    show_interval = crc_train_config['show_interval']
    valid_interval = crc_train_config['valid_interval']
    early_stop = crc_train_config['early_stop']
    checkpoints_dir = crc_train_config['checkpoints_dir']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_loader,valid_loader = train_loader(train_data_path)
    model = make_model('resnet_rnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(reload_checkpoint,map_location=device))
    crnn = FineResnetRnn(model)
    logger.info(f'当前运行环境为device: {device}, config:{crc_train_config}')
    assert crnn
    crnn.to(device)
    optimizer = optim.Adam(crnn.parameters(), lr=lr)
    criterion = CTCLoss()
    criterion.to(device)
    early_num = 0
    val_acc = 0.0
    for epoch in tqdm(range(1, epochs)):
        train(epoch,show_interval,crnn,optimizer,criterion,device,t_loader)
        if epoch%valid_interval == 0:
            val_acc, early_num = valid(epoch, crnn, criterion, device, valid_loader, val_acc, early_num,
                                        checkpoints_dir,
                                        decode_method=crc_train_config['decode_method'],
                                        beam_size=crc_train_config['beam_size'])
    logger.info(" fast train over")
    early_num = 0
    optimizer = optim.Adam(crnn.parameters(), lr=m_lr)
    for epoch in tqdm(range(epochs + 1, epochs + m_epochs + 1)):
        train(epoch, show_interval, crnn, optimizer, criterion, device, t_loader)
        if epoch % valid_interval == 0:
            val_acc, early_num = valid(epoch, crnn, criterion, device, valid_loader, val_acc, early_num,
                                        checkpoints_dir,
                                        decode_method=crc_train_config['decode_method'],
                                        beam_size=crc_train_config['beam_size'])

        if early_num > early_stop:
            logger.info(f"Early Stop in epoch:{epoch}")
            break
    logger.info(" All train over")

if __name__ == '__main__':
    import argparse
    seed = 100
    setup_seed(seed)
    parser = argparse.ArgumentParser(description='Train captcha model')
    parser.add_argument('--train_path', type=str,required= False,default= config.train_data_path, help='The path of train dataset')
    parser.add_argument('--reload_checkpoint',type=str,required=True, help='The pretrain params')
    args = parser.parse_args()
    model_name = 'resnet_rnn'
    init_log(model_name)
    main(args.train_path,model_name,args.reload_checkpoint)

