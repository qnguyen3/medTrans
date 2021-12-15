'''
Adapted from https://github.com/StephenLouis/ISIC_2019/blob/master/Data_Loader.py
with changes to support LightningDataModule
'''
from dataclasses import dataclass
import os
import csv

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torchvision import transforms

from .utils import ISIC2019Dataset, ISIC2019Dataset_CV
from .base_datamodule import BaseKFoldDataModule


class ISIC2019(pl.LightningDataModule):
    """ISIC 2019 Dataset
        Args:
            train_transform: Transformation for train data
            validation_transform: Transformation for validation data
            test_transform: Transformation for test data
            batch_size: Batch size (default: 1)
            num_workers: num_workers for PyTorch DataLoader (default: 0)
            seed: seed for reproducible purpose (default: 1)
            image_size: image size for transformation (default: 224)
    """
    def __init__(self, data_dir: str = None, 
                train_transform: transforms = None,
                validation_transform: transforms = None,
                test_transform: transforms = None,
                batch_size:int = 1,
                num_workers:int = 0,
                seed:int = 1,
                image_size: int = 224):
        super().__init__()

        # Check if dataset is downloaded
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise ValueError('ISIC 2019 does not support downloading')
        # initialization
        self.train_transform = train_transform
        self.validation_transform = validation_transform
        self.test_transform = test_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.image_size = image_size

        self.num_classes = 8

    def setup(self, stage=None):

        if self.train_transform is None:
            self.train_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])
        if self.validation_transform is None:
            self.validation_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])
        if self.test_transform is None:
            self.test_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])


        self.isic2019_train = ISIC2019Dataset(csv_file = os.path.join(self.data_dir, 
                                            'train_isic2019.csv'),data_dir=self.data_dir,
                                            transforms=self.train_transform)
        self.isic2019_val = ISIC2019Dataset(csv_file = os.path.join(self.data_dir, 
                                            'val_isic2019.csv'),data_dir=self.data_dir, 
                                            transforms=self.validation_transform)
        self.isic2019_test = ISIC2019Dataset(csv_file = os.path.join(self.data_dir, 
                                            'test_isic2019.csv'),data_dir=self.data_dir, 
                                            transforms=self.test_transform)

    def prepare_data(self):
        """
        Prepare ISIC 2019 Data
        """
        if os.path.exists(os.path.join(self.data_dir, 'train_isic2019.csv')):
            print("Data is downloaded and verify")
        training_groundtruth_path = os.path.join(self.data_dir, 
                                                'ISIC_2019_Training_GroundTruth.csv')
        self.split_csv(file = training_groundtruth_path)
            

    def train_dataloader(self):
        return DataLoader(self.isic2019_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.isic2019_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.isic2019_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def split_csv(self, file):
        data = []
        a_train_file = os.path.join(self.data_dir, 
                                    'train_isic2019.csv')
        a_val_file = os.path.join(self.data_dir, 
                                    'val_isic2019.csv')
        a_test_file = os.path.join(self.data_dir, 
                                    'test_isic2019.csv')

        np.random.seed(self.seed)
        train_indices = np.random.choice(25331, 20265, replace=False)
        val_test = np.array(list(set(range(25331)) - set(train_indices)))
        val_indices, test_indices = np.split(val_test, 2)

        with open(file)as afile:
            a_reader = csv.reader(afile)
            labels = next(a_reader)
            for row in a_reader:
                data.append(row)

        if not os.path.exists(a_train_file):
            with open(a_train_file, "w", newline='') as a_train:
                writer = csv.writer(a_train)
                writer.writerows([labels]) 
                writer.writerows(np.array(data)[train_indices])
                a_train.close()

        if not os.path.exists(a_val_file):
            with open(a_val_file, "w", newline='') as a_val:
                writer = csv.writer(a_val)
                writer.writerows([labels]) 
                writer.writerows(np.array(data)[val_indices])
                a_val.close()

        if not os.path.exists(a_test_file):
            with open(a_test_file, "w", newline='')as a_test:
                writer = csv.writer(a_test)
                writer.writerows([labels]) 
                writer.writerows(np.array(data)[test_indices])
                a_test.close()

'''
KFold Implementation
References: https://github.com/PyTorchLightning/pytorch-lightning
            /blob/2faaf35b91a00aff397609a875a66c87f8ed6390/pl_examples/loop_examples/kfold.py
'''
@dataclass
class ISIC2019_CV(BaseKFoldDataModule):
    def __init__(self, data_dir: str = None,
                train_transform: transforms = None,
                val_transform: transforms = None,
                test_transform: transforms = None,
                batch_size:int = 1,
                num_workers:int = 0,
                seed:int = 1,
                image_size: int = 224):
        super().__init__()

        self.data_dir = data_dir

        self.train_transform = train_transform
        self.validation_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.image_size = image_size
        self.train_fold = None
        self.val_fold = None
        

    def prepare_data(self) -> None:
        self.training_groundtruth_path = os.path.join(self.data_dir, 
                                                'ISIC_2019_Training_GroundTruth.csv')
        if os.path.exists(self.training_groundtruth_path):
            print("Data is downloaded and verified")
        else:
            raise ValueError('Missing "ISIC_2019_Training_GroundTruth.csv". Make sure the data is downloaded correctly')
        
    def setup(self, stage=None):

        if self.train_transform is None:
            self.train_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])
        if self.validation_transform is None:
            self.validation_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])
        if self.test_transform is None:
            self.test_transform = transforms.Compose([transforms.Resize([self.image_size, self.image_size]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])

        self.train_set = ISIC2019Dataset_CV(csv_file=self.training_groundtruth_path, data_dir=self.data_dir, 
                                            transforms=self.train_transform)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_set)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_set, train_indices)
        self.val_fold = Subset(self.train_set, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers) 
        






        
