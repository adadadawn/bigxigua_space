import os
import librosa
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from configparser import ConfigParser
from models import model
import cv2
import numpy as np
from models import funtion
import yaml




class AudioImageDataset(Dataset):
    def __init__(self, audio_folder, image_folder, transform=None):
        self.audio_folder = audio_folder
        self.image_folder = image_folder
        self.transform = transform

        # 获取所有音频文件路径
        self.audio_files = os.listdir(audio_folder)
        self.audio_files = [os.path.join(audio_folder, f) for f in self.audio_files]
        # load YAML file
        with open('config.yaml', 'r') as f:
          config = yaml.load(f, Loader=yaml.FullLoader)
        self.sr = config['mcff']['sr']
        self.n_fft = config['mcff']['n_fft']
        self.n_mfcc = config['mcff']['n_mfcc']
        self.fps = config['mcff']['fps']

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 加载音频
        audio_path = self.audio_files[idx]
         # 提取MFCC特征
        mfcc = funtion.auido_feature_extract(audio_path, int(self.sr), int(self.n_fft),
                                            int(self.n_mfcc), int(self.fps))

        mfcc = torch.from_numpy(mfcc).float()


        # 加载图像
        image_folder = os.path.join(self.image_folder, os.path.splitext(os.path.basename(audio_path))[0])
        image_files = os.listdir(image_folder)
        images = []
        for f in image_files:
            image_path = os.path.join(image_folder, f)
            img = cv2.imread(image_path)
            # print(img.shape,'1')            
            # print(img.shape,'2')
            images.append(img)
        # 转换数据类型
        # audio = torch.from_numpy(audio).float()
        #将列表转为np，方便转为tensor返回
        images = np.array(images)
        # print(images.shape,'3')

        return mfcc, images

if __name__ == '__main__':
    audio_folder = "data/audio"
    image_folder = "data/images"
    transform = torchvision.transforms.ToTensor()  # 将图像转为tensor

    dataset = AudioImageDataset(audio_folder, image_folder, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=torch.utils.data._utils.collate.default_collate)
    for mfcc, images in dataloader:
      images = np.array(images)
      print(mfcc.shape,images.shape)
