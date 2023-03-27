# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Modifications: Yutong Wen
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data2 import LIDC_IDRI





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
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

input_channels = 1 # one for bw images
num_classes = 2 # binary for LIDC
num_filters = [32, 64, 128, 192, 192, 192, 192]


# PATH_mcd = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/mcd_model.pt'
PATH_ssn = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/ssn_model.pt'
# PATH_dpn = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_model.pt'



#testing
# from model.DPN import Unet
# import model.DPN as dpn_model
# model = Unet(input_channels, num_classes, num_filters)
# opt = torch.optim.Adam(model.parameters(), lr=0.0001)
# dpn_model.testing(model, test_loader, opt,PATH_dpn)

from model.SSN import Unet
import model.SSN as ssn_model

model = Unet(input_channels, num_classes, num_filters)
opt = torch.optim.Adam(model.parameters(), lr=0.0001)
ssn_model.testing(model, test_loader, opt,PATH_ssn,'A')


# from model.MCD import Unet
# import model.MCD as mcd_model
# model = Unet(input_channels, num_classes, num_filters)
# opt = torch.optim.Adam(model.parameters(), lr=0.0001)
# mcd_model.testing(model, test_loader, opt,PATH_mcd)











