# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Modifications: Marc Gantenbein
# This software is licensed under the Apache License 2.0
# Modifications: Paul Fischer, Yutong Wen 
import numpy as np
import torch
import torch.nn as nn
import wandb

# from DisU.utils import init_weights
from DisU.torchlayers import ReversibleSequence
from DisU.visualization import visual_train,visual_test
from DisU.losses import RKL_Loss
from model.DPN_compute import compute_eval_loss_without_OOD,compute_train_loss_and_train_without_OOD,compute_eval_loss_with_OOD,compute_train_loss_and_train_with_OOD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True, reversible=False):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        if not reversible:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            self.layers = nn.Sequential(*layers)
        else:
            layers.append(ReversibleSequence(input_dim, output_dim, reversible_depth=3))

            self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True, reversible=False):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim,
                                        output_dim,
                                        initializers,
                                        padding,
                                        pool=False,
                                        reversible=reversible
                                        )

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=False)
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters,
                 initializers=None, apply_last_layer=True, padding=True,
                 reversible=False, training=False, latent_dim=3, no_convs_fcomb=4, beta=1.0):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()
        self.prediction = None

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(
                DownConvBlock(input, output, initializers, padding, pool=pool, reversible=reversible))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding, reversible=reversible))

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)

    def sample(self, testing=True):
        return self.prediction

    def forward(self, x, mask=None, training=True, val=False):
        """
        :param x: image to segment
        :param mask: mask which serves as a dummy argument for the forward method
        :param val:
        :return:
        """
        blocks = []

        for i, down in enumerate(self.contracting_path):
            x = down(x)
            # add groupnorm
            # m = nn.InstanceNorm2d(self.num_filters[i])
            m = nn.GroupNorm(int(self.num_filters[i]/2), self.num_filters[i],affine=False)
            x = m(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])
            # add groupnorm
            # m = nn.InstanceNorm2d(self.num_filters[-i-2])
            m = nn.GroupNorm(int(self.num_filters[-i-2]/2), self.num_filters[-i-2],affine=False)
            x = m(x)

        del blocks

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x = self.last_layer(x)

        self.prediction = x
        return x


    def loss(self, mask, rkl_mask):
        loss,pred_alpha = RKL_Loss(self.prediction,mask,rkl_mask)
        return loss,pred_alpha





def train_model(model, train_loader, eval_loader, optim, epochs=1):
    """
    Trains and evaluates the model. Additionally, training stats can be tracked and best performing models (according to evaluation loss and GED) can be saveed
    params:
        model: torch module. The PHiSeg model to train
        train_loader: torch data loader (for training data)
        eval_loader: torch data loader (for evaluation)
        optim: torch optimizer
        epochs: int. The number of epochs to train
    """
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)
    if use_gpu:
        model.cuda()
    
    # initialize W&B project

    wandb.init(project='unet', entity='yutongw',settings=wandb.Settings(start_method='fork'))

    test_dice = []
    train_dice = []
    train_ged =[]
    test_ged =[]
    Loss = 0
  
    # Training
    for epoch in range(epochs):
        print('Epoch', epoch)
        # train model and compute loss
        train_running_loss,diceT,maskT,seggT,oriT,gedT= compute_train_loss_and_train_with_OOD(train_loader, model, optim, use_gpu=use_gpu)
        # print("train dice",diceT)
        train_dice.append(diceT)
        Loss = np.mean(diceT)
        # train_ged.append(gedT)
        visual_train(seggT,maskT,diceT,train_running_loss)
    # save checkpoint
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_pOOD_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_sOOD_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_p2OOD_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_real_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_real_model_100.pt'
    PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_real_model3.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_instance_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_precision_model.pt'
    # all mask 1
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_g1OOD_model.pt'
    # all mask 2
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_gOOD_model.pt'
    # PATH = '/mnt/qb/work/berens/bep958/Probabilistic-Unet-offical/Code/model/checkpoints/dpn_model.pt'
    torch.save({
        'epoch':epochs,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optim.state_dict(),
        'loss':Loss,
    },PATH)

    print("average training dice:", np.mean(train_dice)) 
    return

def testing(model, eval_loader, optim,PATH):
    test_dice = []
    
    # model
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)
    if use_gpu:
        model.cuda()
    wandb.init(project='unet', entity='yutongw',settings=wandb.Settings(start_method='fork'))

    test_dice = []

       
    eval_running_loss,diceE,maskE,entropyE,dataUE,MIE,seggE,oriE,ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR = compute_eval_loss_with_OOD(eval_loader, model, optim, use_gpu=use_gpu)
        
    test_dice.append(diceE)
    # visual_test(seggE,maskE,diceE,oriE,entropyE,dataUE,MIE,eval_running_loss,ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR)

    print("average testing dice:", np.mean(test_dice)) 
    return
