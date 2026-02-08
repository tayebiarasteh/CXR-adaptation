"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import matplotlib.pyplot as plt
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15








class padchest_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, image_size=224):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'padchest')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "masterlist.csv"), sep=',')

        if image_size == 224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        elif image_size == 336:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed336')
        elif image_size == 504:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed504')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        APhorizview = self.subset_df[self.subset_df['view'] == 'AP_horizontal']
        self.subset_df = pd.concat([PAview, APview, APhorizview], ignore_index=True)
        self.file_path_list = list(self.subset_df['ImageID'])

        self.chosen_labels = ['atelectasis', 'cardiomegaly', 'pleural_effusion', 'pneumonia', 'no_finding']



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]['ImageDir'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, str(subset), self.file_path_list[idx])) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=7), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor