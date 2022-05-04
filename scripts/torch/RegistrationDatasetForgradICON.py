import torch
from torch.utils.data import Dataset
import random


class dataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path, phase="train"):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs are compressed into memory
        :param data_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        :param transform: function,  apply transform on data
        : seg_option: pars,  settings for segmentation task,  None for segmentation task
        : reg_option:  pars, settings for registration task, None for registration task

        """
        img = torch.load(f"{data_path}/lungs_train_2xdown_scaled", map_location='cpu')
        mask = torch.load(f"{data_path}/lungs_seg_train_2xdown_scaled", map_location='cpu')
        self.img = [[(d[0]+1)*m[0], (d[1]+1)*m[1]] for d,m in zip(img, mask)]
        self.name_list = range(len(self.img))

    def __len__(self):
        return len(self.name_list)
        # return len(self.name_list)*500 if len(self.name_list)<200 and self.phase=='train' else len(self.name_list)  #############################3

    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        img_pair = self.img[idx]
        choice = 0 if random.random() >= 0.5 else 1
        

        return img_pair[choice][0], img_pair[1-choice][0], str(self.name_list[idx])
