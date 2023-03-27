# Author: Paul Fischer
# Modification: Yutong Wen
import torch
import numpy as np
from DisU.evaluations import generalised_energy_distance_unet_batch, DSC_score_batch
def compute_train_loss_and_train(train_loader, model, optimizer, use_gpu):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0

    dice_coeff = []
    ged = []

    for x,y,_ in train_loader:
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        
        # forward pass
        outputs = model(x)
        batch_size = x.shape[0]
        images = x[-1]
        mask = y[-1]
        

        # compute loss 

        loss, segg= model.loss(y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track loss 
        running_loss += loss
        dice_ls = DSC_score_batch(batch_size,y,segg)
        # ged.append(generalised_energy_distance_unet_batch(batch_size,y,segg))
        dice_coeff.append(np.mean(dice_ls))
    
    epoch_loss = running_loss # /12078
    
    return epoch_loss,dice_coeff,segg[-1],mask,images

def compute_eval_loss(test_loader, model, optimizer, use_gpu):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    model.eval()

    dice_coeff = []
    ged = []

    running_loss = 0.0
    with torch.no_grad():
        for x,y,_ in test_loader:
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            
            # forward pass
            outputs = model(x)
            images = x[-1]
            batch_size = x.shape[0]
            mask = y[-1]

            # compute loss 
            loss,segg = model.loss(y)
            

            # track loss 
            running_loss += loss
            dice_ls = DSC_score_batch(batch_size,y,segg)
            # ged.append(generalised_energy_distance_unet_batch(batch_size,y,segg))
            dice_coeff.append(np.mean(dice_ls))
       
    epoch_loss = running_loss # /1509
    
    return epoch_loss,dice_coeff,segg[-1],mask,images