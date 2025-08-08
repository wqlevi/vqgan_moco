"""
How data blobs passed through:
    PIL -> Tensor -> numpy
"""
from glob import glob
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import os
from PIL import Image

#import jax.numpy as jnp
from torch.utils import data
from torch import Tensor
#from jax.tree_util import tree_map
from torchvision.io import read_image
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np


class MoCo(data.Dataset):
    def __init__(self, img_dir:str, csv_file:str, image_size:int=256):
            self.img_dir = img_dir
            self.csv_file = csv_file
            self.files = pd.read_csv(csv_file)

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                #transforms.Normalize((13.1823,), (21.1146,)),
                transforms.Lambda(lambda x: (x - x.min())/ (x.max() - x.min())),  # Normalize to [0, 1]
            ])
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.img_dir, self.files.iloc[idx,0])
        image = Image.open(filename)
        image = self.transform(image)
        return image

def load_moco(
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
):
    dataloader = data.DataLoader(
        MoCo(
            img_dir="/workspace/working_buffer/0/sim/0",
            csv_file="./files_no_bg.csv",
            image_size=image_size
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
