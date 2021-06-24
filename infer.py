import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import sys
from network import Generator
from utils import show_tensor_images
from sys import argv
face_location = argv[1]
anime_location = argv[2]

import time 

pretrained_epochs = 49400
load_shape = 256
target_shape = 256
dim_A = 3
dim_B = 3
device = 'cpu'
transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.CenterCrop(load_shape),
    transforms.ToTensor(),
])
timeit = time.time()
gen_AB = Generator(dim_A, dim_B).to(device)
gen_BA = Generator(dim_B, dim_A).to(device)


pre_dict = torch.load(f'./models/cycleGAN_{pretrained_epochs}.pth', map_location=torch.device('cpu'))
gen_AB.load_state_dict(pre_dict['gen_AB'])
gen_BA.load_state_dict(pre_dict['gen_BA'])


face = transform(Image.open(face_location)).to(device)

anime = transform(Image.open(anime_location)).to(device)
face = (face - 0.5) * 2
anime = (anime - 0.5) * 2
face = torch.unsqueeze(face[:3], 0)
anime = torch.unsqueeze(anime, 0)
face = nn.functional.interpolate(face, size=target_shape)
anime = nn.functional.interpolate(anime, size=target_shape)


with torch.no_grad():
	fake_face = gen_AB(face)
	fake_anime = gen_BA(anime)

print(time.time() - timeit)
show_tensor_images(torch.cat([fake_face, face]), size=(dim_B, target_shape, target_shape))

show_tensor_images(torch.cat([fake_anime, anime]), size=(dim_A, target_shape, target_shape))
