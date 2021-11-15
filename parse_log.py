#-*- coding: utf-8 -*-
import os
from tqdm import  tqdm
import matplotlib.pyplot as plt
import  numpy as np

def parse_log(path):
    train_loss,train_epoch= [],[]
    valid_loss,valid_epoch,valid_acc = [],[],[]
    with open(path,'r',encoding='gbk') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            print(line)
            item = str(line)
            train = item.find("-INFO:Train epoch")
            valid = item.find("-INFO:Valid epoch")
            if train != -1:
                print(item)
                item = item.split('-INFO:')[1]
                epoch_info, loss_info = item.split(",")
                #print(epoch_info,loss_info)
                epoch = int(epoch_info.split(':')[1])
                loss = float(loss_info.split(':')[1])
                train_loss.append(loss)
                train_epoch.append(epoch)
            elif valid != -1:
                print(item)
                item = item.split('-INFO:')[1]
                epoch_info, loss_info,acc_info = item.split(",")
                epoch = int(epoch_info.split(':')[1])
                loss = float(loss_info.split(':')[1])
                acc = float(acc_info.split(':')[1])
                valid_epoch.append(epoch)
                valid_loss.append(loss)
                valid_acc.append(acc)

    print(max(valid_acc))
    return train_loss,train_epoch,valid_loss,valid_epoch,valid_acc


def plt_show(path):
    train_loss, train_epoch, valid_loss, valid_epoch, valid_acc = parse_log(path)
    log_name = path.split('.')[-2]
    plt.subplot(211)
    plt.plot(train_epoch,train_loss,c = 'r',label = 'train loss')
    plt.plot(valid_epoch,valid_loss,c = 'b',label = 'valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss figure')
    plt.legend()
    plt.subplot(212)
    plt.plot(valid_epoch, valid_acc,c = 'g',label = 'valid acc')
    plt.xlabel('epoch')
    plt.ylabel('valid acc')
    plt.title('acc figure')
    plt.legend()
    plt.savefig(f"{log_name}_loss_acc.png")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path ="checkpoints/crc_checkpoints/20211020/20211020_3371_resnet_rnn.pt"
    plt_show(path)