import numpy as np 
import os
import random 
from PIL import Image  
import torch 
from torch.utils.data import Dataset

from data.transforms import ApplyTransform

class ImageDataset(Dataset):
    def __init__(self, parameters, paths, annotations, species_info, batch_size=1, transform=False, shuffle=False, valid=False):
        self.parameters = parameters
        self.paths = np.array(paths)
        self.species_info = np.array(species_info)
        self.batch_size = batch_size
        self.transform = transform
        self.order = np.array([i for i in range(len(self.paths))])
        self.shuffle = shuffle
        self.Transfrom_object = ApplyTransform(self.parameters["model"], self.parameters["img_size"]) 
        self.valid = valid
        self.labels = np.array(annotations)
        self.num_batches = len(self.labels)//self.batch_size


    def __len__(self):
        return self.num_batches
    
    def load_img(self,im_path):
        im = Image.open(os.path.join(self.parameters["path_source_img"],im_path)).convert('RGB')
        return Image.fromarray(np.array(im)[:, :, ::-1])
    

    def __getitem__(self, idx):
        order = self.order[idx*self.batch_size:(idx*self.batch_size)+self.batch_size]
        images = [self.load_img(p) for p in self.paths[np.array(order)]]
        images = [self.Transfrom_object.resize(im) for im in images]
        images = [self.Transfrom_object.augment(im) for im in images]
        images = [self.Transfrom_object.normalise(im).unsqueeze(0) for im in images]
        labels = self.labels[order]
        species_info = self.species_info[order]
        paths = np.array(order)
        return torch.stack(images), torch.tensor(labels, dtype=torch.float), paths, species_info

   