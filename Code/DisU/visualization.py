# Author: Yutong Wen
import wandb
import matplotlib as mpl
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
import matplotlib.pyplot as plt

def visual_train(train_segmentation,train_gT,train_dice,train_loss):
	segmentation = train_segmentation.cpu().detach().numpy() # greyscale
	gT = train_gT.cpu().detach().numpy() #black and white

	return wandb.log({
            'Train Loss': train_loss,
            'Train Dice' : np.mean(train_dice),
            # 'Train GT': wandb.Image(gT),   
            'Train Segmentation':wandb.Image(segmentation)

        })

# def visual_test(test_segmentation,test_gT,test_dice,test_image, test_entropy,test_dataU,test_MI,test_loss,ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR):
def visual_test(test_segmentation,test_gT,test_dice,test_image, test_entropy,test_dataU,test_MI,print):

	segmentation = test_segmentation.cpu().detach().numpy() # greyscale
	image = test_image.cpu().detach().numpy() #black and white
	gT = test_gT.cpu().detach().numpy() #black and white

	# cm_color = mpl.cm.get_cmap('YlGnBu')
	cm_color = mpl.cm.get_cmap('hot')
	entropy = cm_color(test_entropy.cpu().detach().numpy())
	entropy = np.uint8(entropy*255)
	dataU = cm_color(test_dataU.cpu().detach().numpy())
	dataU = np.uint8(dataU*255)
	MI = cm_color(test_MI.cpu().detach().numpy())
	MI = np.uint8(MI *255)
	# segmentation = np.uint8(segmentation *255)
	# if print:

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(image[0],cmap=mpl.cm.get_cmap('gray'))
	plt.axis('off')
	
	plt.savefig('/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/CT/mcdadd3/image'+print+'.pdf',dpi = 200,layout="constrained")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(segmentation,cmap=mpl.cm.get_cmap('gray'))
	plt.axis('off')
	
	plt.savefig('/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/CT/mcdadd3/seg'+print+'.pdf',dpi = 200,layout="constrained")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(entropy, cmap=cm_color)
	fig.colorbar(cax).ax.tick_params(labelsize=24)
	plt.axis('off')

	plt.savefig('/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/CT/mcdadd3/TE'+print+'.pdf',dpi = 200,layout="constrained")



	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(dataU, cmap=cm_color)
	fig.colorbar(cax).ax.tick_params(labelsize=24)
	plt.axis('off')
	
	plt.savefig('/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/CT/mcdadd3/DU'+print+'.pdf',dpi = 200,layout="constrained")

	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(MI, cmap=cm_color)
	fig.colorbar(cax).ax.tick_params(labelsize=24)
	plt.axis('off')

	plt.savefig('/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/CT/mcdadd3/MI'+print+'.pdf',dpi = 200,layout="constrained")



	return wandb.log({
            # 'Total loss': test_loss,
            'Dice' : np.mean(test_dice),
            'Images': wandb.Image(image),
            'GT': wandb.Image(gT),   
            'Segmentation':wandb.Image(segmentation),
            'Total entropy':wandb.Image(entropy),
            'Data Uncertainty':wandb.Image(dataU),
            'MI':wandb.Image(MI)
			# 'Entropy AUROC': ent_AUROC,
        	# 'Expect data uncertainty AUROC' : DU_AUROC,
        	# 'Entropy of MI AUROC:':MI_AUROC,
        	# 'Entropy AUPR': ent_AUPR,
        	# 'Expect data uncertainty AUPR' : DU_AUPR,
			# 'Entropy of MI AUPR:':MI_AUPR

        })



