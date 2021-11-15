# 数据预处理
import logging
import os
import json
from PIL.PyAccess import new
import pandas as pd
import torch
import torchvision.utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import dataset,dataloader
from util import init

class OcrDataset(dataset.Dataset):
    """
    ## 加载数据，数据格式为
    # train: label.png
    # test: index.png
    """

    def __init__(self, root:str, char2labels:dict,logger:logging, multi = False, transformer = None,train = True):
        """
        captcha dataset
        :param root: the paths of dataset, 数据类型为 root/label.png ...
        :param transformer: transformer for image
        :param train: train of not
        """
        super(OcrDataset, self).__init__()
        assert root and char2labels
        self.char2labels = char2labels
        self.root = root
        self.train = train
        self.transformer = transformer
        self.labels = None
        self.logger = logger
        paths = [os.path.join(self.root,path) for path in os.listdir(self.root)]
        self._extract_images(paths)
        # if self.train:
        #     self._check_images()

    
    def _extract_img_paths(self,path):
        imgs = []
        labels = []
        with open(path,encoding='utf-8') as f:
            items = json.loads(f.read()).items()
            for k , v in items:
                imgs.append(k)
                labels.append(v)
            img_paths = [f'images/{img}' for img in imgs]
        return img_paths,labels

    def _extract_images(self,paths):
        self.image_paths = []
        self.labels = []
        for path in paths:
            self.logger.info(path)
            if not os.path.isdir(path):
                continue
            if self.train:
                img_paths,labels = self._extract_img_paths(os.path.join(path,'gt.json'))
                self.labels.extend(labels)
                img_paths = [os.path.join(path,img_path) for img_path in img_paths]
                self.image_paths.extend(img_paths)
            else:
                images_paths = os.listdir(os.path.join(path,'images'))
                img_paths = list(filter(lambda x: x.endswith('.png') or x.endswith('.jpeg') or x.endswith('.jpg'), images_paths))
                self.labels.extend(img_paths)
                img_paths = [os.path.join(path,f'images/{img}') for img in img_paths]
                self.image_paths.extend(img_paths)
        
        assert len(self.image_paths) == len(self.labels) 
    


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            img.load() # required for png.split()
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=a) # 3 is the alpha channel
            img  =  background
        #print(img.size)
        if self.transformer:
            img = self.transformer(img)
        # print(img.size)
        if self.train:
            label = str(self.labels[idx])
            target = [self.char2labels[c] for c in label]
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return img, target, target_length
        else:
            return img
   
        
   
      
 

def resizeNormalize(image,imgH, imgW,mean,std,train = False):
    """
    resize and normalize image
    """
    if train:
        transformer = transforms.Compose(
        [
         transforms.RandomAffine((0.9,1.1)),
         transforms.RandomRotation(3),
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    else:
        transformer = transforms.Compose(
        [
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    return transformer(image)


class OcrCollateFn(object):

    def __init__(self,imgH=32, imgW=100, keep_ratio=False,mode = 'Train', mean = (0.485, 0.456, 0.406), std =  (0.229, 0.224, 0.225)) -> None:
        """
        mode: 模式 Train, Test, Valid
        """
        super().__init__()
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.mode = mode
        self.mean = mean
        self.std = std
    
    def resize(self,img:Image.Image):
        w, h = img.size
        ratio = h/float(self.imgH)
        w_ = int(w/ratio)
        if w_ > self.imgW:
            w_ = self.imgW
            img = img.resize((w_,self.imgH))
        elif w_ < self.imgW:
            img = img.resize((w_,self.imgH))
            new_img = Image.new('RGB',(self.imgW,self.imgH),color=(255,255,255))
            new_img.paste(img,((self.imgW - w_)//2, 0))
            img = new_img
        return img

    def __call__(self, batch):
        if self.mode == 'Test':
            images = batch
        else:
            images, targets, target_lengths = zip(*batch)
        if self.keep_ratio:
            max_ratio = 0.0
            for image in images:
                w,h = image.size
                max_ratio = max(max_ratio,w/float(h))
            self.imgW = int(max_ratio*self.imgH + 0.5)
            #self.imgH = int(max_ratio*self.imgH)
        # print(images)=
        images = [self.resize(image) for image in images]
        images = [resizeNormalize(image,self.imgH,self.imgW,self.mean,self.std,self.mode == 'Train') for image in images]
        # print(images[0].shape)
        images = torch.stack(images, 0)
        if self.mode == 'Test':
            return images
        else:
            targets = torch.cat(targets, 0)
            target_lengths = torch.cat(target_lengths, 0)
            return images, targets, target_lengths
        
    

def train_loader(config,logger,transformer = None):
    """
    
    :param train_path:  the path of training data
    :param batch_size: 
    :param height resize height
    :param width: resize width
    :return: 
    """""
    # if transformer is None:
    #     transformer = transforms.Compose(
    #         [
    #           #transforms.RandomAffine((0.9,1.1)),
    #           #transforms.RandomRotation(8),
    #           transforms.Resize((height, width)),
    #           transforms.ToTensor(),
    #           transforms.Normalize(mean=config.mean,std= config.std)
    #          ]
    #     )
    train_set = OcrDataset(config['train']['input_path'],char2labels = config['base']['char2labels'],logger = logger,
                multi = config['train']['multi'], transformer=transformer)
    train_len = int(len(train_set)*config['base']['train_rate'])
    train_data, val_data = torch.utils.data.random_split(train_set,[train_len,len(train_set)-train_len])
    return dataloader.DataLoader(train_data, batch_size=config['train']['batch_size'], shuffle=True,
           collate_fn= OcrCollateFn(config['base']['height'],config['base']['width'],config['train']['keep_ratio'],'Train',config['base']['mean'],config['base']['std'])),\
           dataloader.DataLoader(val_data, batch_size=config['train']['batch_size'], shuffle=True,
           collate_fn= OcrCollateFn(config['base']['height'],config['base']['width'],config['train']['keep_ratio'],"Valid",config['base']['mean'],config['base']['std']))


def test_loader(config:dict,logger:logging,transformer = None):
    """

    :param test_path:
    :param batch_size:
    :param x: resize
    :param y:
    :return:
    """
    # if transformer is None:
    #     transformer = transforms.Compose(
    #     [transforms.Resize((height, width)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=config.mean, std=config.std)
    #      ]
    # )
    test_set = OcrDataset(config['test']['input_path'],char2labels = config['base']['char2labels'],logger = logger,
                            multi = config['test']['multi'],train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=config['test']['batch_size'], shuffle=False,collate_fn = 
               OcrCollateFn(config['base']['height'],config['base']['width'],config['test']['keep_ratio'],'Test',
               config['base']['mean'],config['base']['std']))



if __name__ == '__main__':
     height,width = 32,100
    #  transformer = transforms.Compose(
    #     [
    #         #transforms.RandomAffine((0.9, 1.1)),
    #         #transforms.RandomRotation(8),
    #         transforms.Resize((32, int(width/(height/3)))),
    #         transforms.ToTensor(),
    #     ]
    #  )
     config, logger = init(100,'configs/local/resnet_v2_rnn_ctc.yaml',save_log=True)
     train_loade,val_loader = train_loader(config,logger,transformer = None)
     imgs, targets, target_lens  = next(iter(train_loade))
     grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     plt.imshow(grid_img.permute(1, 2, 0))
     plt.imsave(f"pres/preprocessed_{height}_{width}.jpg",grid_img.permute(1, 2, 0).numpy())
     # num = 0
     # for imgs, targets, target_lens  in train_loader:
     #     num += len(imgs)
     #     logger.info(f"imgs:{imgs.shape}, {num}")



