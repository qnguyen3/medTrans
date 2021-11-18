from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import csv
import os
import torch

class ISIC2019Dataset(Dataset):
    def __init__(self, csv_file: str, 
                data_dir: str, 
                transforms: transforms=None):
        
        self.images = read_labels_csv(csv_file)
        self.data_dir = os.path.join(data_dir, 'ISIC_2019_Training_Input')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, target = self.images[index]
        image = Image.open(os.path.join(self.data_dir,image_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = np.argmax(target)

        return image, target

def read_labels_csv(file,header=True):
    images = []
    num_categories = 0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images