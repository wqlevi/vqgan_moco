'''
TODOS:
    - [x] save all data in ful data format (fp32) not PNG (uint8)
    - [x] input PNG in uint8 [0, 255], and is transformed into float32 [0, 1]
    - [x] save_image to save value between 0 and 1 for grayscale
    - [x] make dataloader iter faster (now 0.3 s/iter)
'''
import enum
import os
import torch
from typing import List
import random
from PIL import Image
import pandas as pd
import h5py
import nibabel as nb

import numpy as np
import jax.numpy as jnp
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, RandomCrop
from torchvision.io import read_image
from typing import List, Tuple
from glob import glob
from pathlib import Path
import torchvision.transforms.functional as TF

from torchvision.utils import save_image
import timeit, time

from prefetch_generator import BackgroundGenerator
from utils import cast_to_int8, make_grid_grayimg

norm = lambda x: (x - x.min())*255/(x.max() - x.min())
class DataLoaderX(DataLoader):
    #def __init__(self, dataset, **kwargs):
        #super().__init__()
        #self.gen = BackgroundGenerator(dataset)
    #def __iter__(self):
        #return self.gen.__iter__()
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader

def gen_loader(dataloader):
    while True:
        yield from dataloader

class Cast_jnp(object):
    """
    dataloader transformation fn:
        1. permute to JAX: [B,H,W,C] or torch.Tensor: [B,C,H,W];
        2. cast img to numpy.NdArray or torch.Tensor
    """
    def __init__(self, torch_tensor:bool=False, norm_type:str='z-score'):
        self.torch_tensor = torch_tensor
        # mean and std for UKB70k abdonimal dataset
        mean_data:float=13.1823
        std_data:float=21.1146
        #self.normalize = lambda x: (x - mean_data) / std_data if norm_type == 'z-score' else (x-x.min())*1/(x.max() - x.min())
        self.normalize = lambda x: (x - mean_data) / std_data if norm_type == 'z-score' else x/255 # zscore | 0-1 norm
    def __call__(self, img:Tensor): #[CHW]
        img = img.permute(1,2,0) if not self.torch_tensor else img # torch.Tensor [CHW] | JAX.Tensor [HWC]
        if not self.torch_tensor:  self.arr = np.array(img, dtype=jnp.float32)
        self.arr = img.float()
        self.arr = self.normalize(self.arr)
        return self.arr

class MyDset(Dataset):
    """
        1. read csv and load image from the csv entries;
        2. apply random crop
    """
    def __init__(self, csv_file, img_dir, crop_size:Tuple[int, int]=True, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.files = pd.read_csv(csv_file) # only file names
            self.crop_size_h, self.crop_size_w = crop_size
            self.random_crop = RandomCrop(crop_size)
            self.shape:Tuple = None

    def __len__(self):
        return len(self.files)
    #@profile
    def __getitem__(self, idx):
        gt_filename = Path(self.img_dir).joinpath(self.files.iloc[idx,0])
        image = read_image(gt_filename)
        image = self.random_crop(image)
        image = self.transform(image)
        self.shape = image.shape
        return {'gt':image}


class MyDset_csv_cached(Dataset):
    def __init__(self, csv_file, img_dir, sim_img_dir,  crop_size:Tuple[int, int]=True, transform=None):
            self.img_dir = img_dir
            self.sim_img_dir = sim_img_dir
            self.transform = transform
            self.files = pd.read_csv(csv_file) # only file names
            self.crop_size_h, self.crop_size_w = crop_size

            self.labels:List = os.listdir(sim_img_dir) # 10-1=9 labels
            self.labels.remove('0') 
            self.cache = SharedCache(
                    size_limit_gib=32,
                    dataset_len = 75500,
                    data_dims= (224, 168),
                    dtype=torch.uint8
                    )

    def _randomcrop(self, img, sim):
        i,j,h,w = RandomCrop.get_params(
                img, output_size=(self.crop_size_h,
                                  self.crop_size_w)
                )
        img, sim = [TF.crop(x, i, j, h, w) for x in [img, sim]]
        return img, sim

    def __len__(self):
        return len(self.files)
    #@profile
    def __getitem__(self, idx): # 3 sec per bs=128 num_worker=4
        image = self.cache.get_slot(idx)
        if image is None:
            gt_filename = Path(self.img_dir).joinpath(self.files.iloc[idx,0])
            image = read_image(gt_filename)
            self.cache.set_slot(idx, image)
        #image = read_image(gt_filename) 
        labels = self.labels
        label = random.choice(labels)
        sim_file_name = Path(self.sim_img_dir) / str(label) / "sim"/ str(label) # NOTE: slower than str concat
        sim_file_name = sim_file_name.joinpath(self.files.iloc[idx,0])

        image_sim = read_image(sim_file_name)
        image, image_sim = self._randomcrop(image, image_sim)
        image = self.transform(image) # NOTE: fp32, [0, 1]
        labels = self.labels
        image_sim = self.transform(image_sim)
        return {'gt':image, 'noisy':image_sim, 'label':int(label)}

class MyDset_h5(Dataset):
    def __init__(self, img_dir, orientations:List = [0,1,2], crop_size:Tuple[int, int]=[128, 128], transform=None, pin_mem:bool=False, data_key:str='images'):
            self.img_dir = img_dir
            self.transform = transform
            self.crop_size_h, self.crop_size_w = crop_size
            self.pin_mem = pin_mem
            self.orientations = orientations
            self.random_crop = RandomCrop(crop_size)
            self.data_key = data_key

            with  h5py.File(self.img_dir, 'r') as f:
                self.vol_shape = f[data_key][0].shape # [H,W,D]
                self.files_len = len(f[data_key]) 

                self.slices_per_orientation={}
                for dim in self.orientations: # dict{0|1|2 : shape[0|1|2]}
                    self.slices_per_orientation[dim] = self.vol_shape[dim]
                self.total_slices = self.files_len * sum(self.slices_per_orientation.values())

                self.flat_index_mapping = [(vol_idx, orient, slice_idx) for vol_idx in range(self.files_len) for orient in self.orientations for slice_idx in range(self.slices_per_orientation[orient])]


                if self.pin_mem: 
                    self.data = torch.Tensor(f[self.data_key][:]) # now loading 3D

    def _randomcrop(self, img, sim):
        i,j,h,w = RandomCrop.get_params(
                img, output_size=(self.crop_size_h,
                                  self.crop_size_w)
                )
        img, sim = [TF.crop(x, i, j, h, w) for x in [img, sim]]
        return img, sim

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx): # 
        vol_idx, orient, slice_idx = self.flat_index_mapping[idx]
        if self.pin_mem:
            data = self.data[vol_idx]
        else:
            with h5py.File(self.img_dir, 'r') as f:
                data = torch.tensor(f[self.data_key][vol_idx]) # 3D vol

        data = data.narrow(orient, slice_idx, 1).squeeze()[None,...]# slicing 3D vol into 2D with [C, H(W, D), W(H,D)] 
        data = self.random_crop(data)
        return data
        
class MyDset_csv(Dataset):
    def __init__(self, csv_file, img_dir, sim_img_dir,  crop_size:Tuple[int, int]=True, transform=None):
            self.img_dir = img_dir
            self.sim_img_dir = sim_img_dir
            self.transform = transform
            self.files = pd.read_csv(csv_file) # only file names
            self.crop_size_h, self.crop_size_w = crop_size

            self.labels:List = os.listdir(sim_img_dir) # 10-1=9 labels
            self.labels.remove('0') 

    def _randomcrop(self, img, sim):
        i,j,h,w = RandomCrop.get_params(
                img, output_size=(self.crop_size_h,
                                  self.crop_size_w)
                )
        img, sim = [TF.crop(x, i, j, h, w) for x in [img, sim]]
        return img, sim

    def __len__(self):
        return len(self.files)
    #@profile
    def __getitem__(self, idx): # 0.048 sec per bs=128 num_worker=4
        gt_filename = Path(self.img_dir).joinpath(self.files.iloc[idx,0])
        image = read_image(gt_filename) 

        
        labels = self.labels # random select a level of motion
        label = random.choice(labels)
        sim_file_name = Path(self.sim_img_dir) / str(label) / "sim"/ str(label) # NOTE: slower than str concat
        sim_file_name = sim_file_name.joinpath(self.files.iloc[idx,0])

        image_sim = read_image(sim_file_name)
        image, image_sim = self._randomcrop(image, image_sim)
        image = self.transform(image) # NOTE: fp32, [0, 1]
        image_sim = self.transform(image_sim)
        return {'gt':image, 'noisy':image_sim, 'label':int(label)}
#@profile
def run_h5loader():
    img_path = '/home/rawangq1/git_wq/vqvae2_mine/datasets/dataset.5h'
    dataset = MyDset_h5(img_path, crop_size=(128, 128), pin_mem = True,transform=Cast_jnp(torch_tensor=True, norm_type='zero-one'))

    test = dataset[0] # debugging, mimicking __getitem__

    #dataloader = setup_loader(dataset, 128)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
    print("len of dataset {}".format(len(dataset)))
    #for batch in enumerate(dataloader): # 1e7 * 1e-6 sec
    for _ in range(100):
        batch = next(iter(dataloader))
        x = batch
        print(x.shape)
#@profile
def run_dataloader():
    gt_path = '/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d/0/sim/0'
    sim_path = '/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d/'
    dataset = MyDset_csv('/home/rawangq1/git_wq/SSL_veronika/files.csv',gt_path, sim_path, crop_size=(128, 128), transform=Cast_jnp(torch_tensor=True, norm_type='zero-one'))

    test = dataset[0] # debugging, mimicking __getitem__

    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)
    print("len of dataset {}".format(len(dataset)))
    #for batch in enumerate(dataloader): # 1e7 * 1e-6 sec
    for _ in range(100):
        batch = next(iter(dataloader))
        x = batch
#@profile
def run_prefetch():
    from torchvision.utils import save_image
    gt_path = '/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d/0/sim'
    sim_path = '/mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/2d/'
    #dataset = MyDset_csv_cached(, gt_path, sim_path, crop_size=(128,128), transform=Cast_jnp(torch_tensor=True, norm_type='zero-one'))
    #dataset = MyDset('/home/rawangq1/git_wq/SSL_veronika/files.csv',gt_path, crop_size=(128, 128), transform=Cast_jnp(torch_tensor=True, norm_type='zero-one'))
    dataset = MyDset_csv('/home/rawangq1/git_wq/SSL_veronika/subset_file.csv',gt_path, sim_path, crop_size=(128, 128), transform=Cast_jnp(torch_tensor=True, norm_type='z-score'))

    test = dataset[0] # debugging, mimicking __getitem__

    dataloader = setup_loader(dataset, 128)
    #loader_gen = gen_loader(dataloader)

    print("len of dataset {}".format(len(dataset)))
    #for idx in range(10):
        #batch = next(loader_gen)
    for i, batch in enumerate(dataloader):
        x = batch['gt'] # [128,1,128,128]
        #x = norm(x.mul(21.1146).add_(13.1823)).to(torch.uint8)
    #grid = make_grid_grayimg(x)
    #im = Image.fromarray(cast_to_int8(grid).squeeze().numpy()).convert('L')
    #im.save('test_im_grid.png')
    #save_image(torch.cat(ls), 'test_io.png')

if __name__ == '__main__':
    run_h5loader()
    #run_prefetch()
    run_dataloader()
