import torch
import numpy as np
from torch.utils.data import DataLoader
from load_LIDC_data2 import LIDC_IDRI
import wandb


seed = 66
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = 'data/')
DB = DataLoader(dataset, batch_size=1)
wandb.init(project='unet', entity='yutongw',settings=wandb.Settings(start_method='fork'))

i = 0
j = 0 # all four experts mark nodule >= 3mm
m = 0 # three out of four experts
k = 0 # 2 out of 4
l = 0 # only one
for x,y,_ in DB:


	i = i + 1
	print(x.shape)
	print(y.shape)
	# numx =30
	# numy = 30
	# x[:,0,numx:numx+35,numy:numy+35] = 1
	# y[:,:,numx:numx+35,numy:numy+35] = 0
	# wandb.log({
    #         'image' : wandb.Image(x[-1][0].numpy()),
    #         'an1': wandb.Image(y[-1][0].numpy()),
    #         'an2': wandb.Image(y[-1][1].numpy()),
	# 		'an3': wandb.Image(y[-1][2].numpy()),
	# 		'an4': wandb.Image(y[-1][3].numpy())})
	# if i > 20:
	# 	break
	# print(y.shape) #[4,128,128]

	if (1 in y[0][0].long()) and (1 in y[0][1].long()) and (1 in y[0][2].long()) and (1 in y[0][3].long()):
				j = j + 1
	elif int(1 in y[0][0].long()) + int (1 in y[0][1].long()) +int (1 in y[0][2].long()) + int (1 in y[0][3].long()) == 3:
				m = m + 1
	elif int(1 in y[0][0].long()) + int (1 in y[0][1].long()) +int (1 in y[0][2].long()) + int (1 in y[0][3].long()) == 2:
				k = k + 1
	elif int(1 in y[0][0].long()) + int (1 in y[0][1].long()) +int (1 in y[0][2].long()) + int (1 in y[0][3].long()) == 1:
				l = l + 1

print("number of the images:", i)
print("number of 4 experts annotation:",j)
print("number of 3 experts annotation:",m)
print("number of 2 experts annotation:",k)
print("number of 1 experts annotation:",l)
	# im = Image.fromarray(x[0][0].numpy())
	# im = im.astype(np.uint8)
	# im.save("/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/imagedata/" +str(i)+".png")
	
	# mask = Image.fromarray(y.numpy())
	# mask = mask.astype(np.uint8)
	# mask.save("/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/imagedata/mask-" +str(i)+".png")
	# i = i + 1
	# break
