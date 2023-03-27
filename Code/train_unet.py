# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/


import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI


from model.unet import Unet
import model.unet as unet_model

seed = 66
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=3, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))


# define the optimizer you want to use 

# epochs = 1

# train the model for 50 epochs and log with W&B; wandb_project is optional as is save_model and save_path 
# define the UNet model
input_channels = 1 # one for bw images
num_classes = 2 # binary for LIDC
num_filters = [32, 64, 128, 192, 192, 192, 192]
model = Unet(input_channels, num_classes, num_filters)
opt = torch.optim.Adam(model.parameters(), lr=0.000001)
unet_model.train_model(model, train_loader, test_loader, opt, epochs=1)#
