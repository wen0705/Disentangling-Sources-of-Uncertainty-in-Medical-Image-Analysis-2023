# Author: Yutong Wen
import numpy as np
import torch
import torch.nn as nn
from DisU.evaluations import DSC_score_batch,misclassification_scores_MC,OODdetection_scores_MC,OODdetection_scores_MC2
from DisU.visualization import visual_test
import random
from sklearn.metrics import roc_auc_score,average_precision_score
import wandb
import matplotlib as mpl
from model.OOD_generator import OOD_surround,OOD_partial_mask2,OOD_partial_mask1,OOD_all_mask1,OOD_all_mask2
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math

def compute_train_loss_and_train_without_OOD(train_loader, model, optimizer, use_gpu):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0

    dice_coeff = []

    for x,y,_ in train_loader:
        num = random.randint(10,80)
        # x[-1][0][num:num+30,num:num+30] = 1
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        
        # forward pass
        outputs = model(x)

        batch_size = x.shape[0]
        mask = y[-1]
        images =x[-1]

        # compute loss 
        loss,prediction = model.loss(y)
        _, segg = torch.max(prediction, dim=1)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

        # track loss 
        running_loss += loss
        dice_ls = DSC_score_batch(batch_size,y,segg)
        dice_coeff.append(np.mean(dice_ls))
    
    epoch_loss = running_loss # /12078

    return epoch_loss,np.mean(dice_coeff),segg[-1],mask,images

def compute_eval_loss_without_OOD(test_loader, model, optimizer, use_gpu):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    # model.eval()

    dice_coeff = []
    ent_score=[]
    du_score =[]
    mi_score =[]
    lbls = []
    ent_AUROC = 0
    DU_AUROC = 0
    MI_AUROC = 0
    ent_AUPR = 0
    DU_AUPR = 0
    MI_AUPR = 0
    ged = []
    num = 0
    running_loss = 0.0
    with torch.no_grad():
        for x_ori,y,_ in test_loader:
        # for x,y,_ in test_loader:
            
            xs = torch.cat((x_ori, x_ori, x_ori,x_ori), 1)
            x = xs.reshape(4,1,128,128)
            y = y.reshape(4,128,128)
            # z = y.clone()
            # b = x_ori.shape[0]
            # xs = torch.cat((x_ori, x_ori, x_ori,x_ori), 1)
            # x = xs.reshape(b*4,1,128,128)
            # y = y.reshape(b*4,128,128)
            z = y.clone()
            
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
            
            # forward pass
            
            
            outputs = model(x)
            images = x[-1]

            # compute loss 
            N = 1114
            # N = 3
            loss,prediction= model.loss(y)
            _, segmentation = torch.max(prediction, dim=1)
            segmentation =  segmentation.expand((1,) + segmentation.shape)
            prediction = prediction.expand((1,) + prediction.shape)
            batch_size = x.shape[0]
            
            for i in range(N):
                outputs = model(x)
                loss,pred= model.loss(y)
                _, segg = torch.max(pred, dim=1)
                pred = pred.expand((1,) + pred.shape)
                dice_ls = DSC_score_batch(batch_size,y,segg)
                dice_coeff.append(np.mean(dice_ls))
                segg = segg.expand((1,) + segg.shape)
                segmentation = torch.cat((segmentation,segg))
                prediction = torch.cat((prediction,pred))

            # track loss 
            running_loss += loss
            # ged.append(generalised_energy_distance_MCD(segmentation,y,N+1))
            
            segmentation = segmentation.float()
            segmentation = segmentation.mean(dim=0) #[batch,128,128]


            # Uncertainty Estimation
            # prediction [1115,2,2,128,128]
            softmax = nn.Softmax(dim = 2)
            prediction = softmax(prediction)
            relu = nn.ReLU(inplace=False)
            prediction = relu(prediction)+1e-6
            
            prediction = prediction.permute(1,0,2,3,4) #[batch,N,2,128,128]
            # to make sure there is no infinity value at uncertainty estimation
            
            prob_tot= prediction.mean(dim=1) #[batch,2,128,128]
            
            
            entropy = -torch.sum(prob_tot*torch.log(prob_tot), dim=1) #[batch,128,128]
            

            data_uncertainty = -torch.sum(prediction * torch.log(prediction),dim=2) #[batch,N,128,128]
            data_uncertainty = data_uncertainty.mean(dim=1)#[batch,128,128]
            # print("du",data_uncertainty)
            MI = entropy-data_uncertainty #[batch,128,128]

            
            
            # AUROC,AUPR for classification
            entropy_score,dataU_score,MI_score,labels = misclassification_scores_MC(batch_size,entropy,data_uncertainty,MI,segmentation,y)

            lbl=list(labels.detach().cpu().numpy())
            du_scores= list(dataU_score.detach().cpu().numpy())
            ent_scores =list(entropy_score.detach().cpu().numpy())
            mi_scores = list(MI_score.detach().cpu().numpy())


            ent_score.append(ent_scores)
            du_score.append(du_scores)
            mi_score.append(mi_scores)
            lbls.append(lbl)
            num = num + 1
            
    

    lbl = list(np.concatenate(lbls).flat)
    ent_scores =list(np.concatenate(ent_score).flat)
    du_scores =list(np.concatenate(du_score).flat)
    mi_scores =list(np.concatenate(mi_score).flat)
    print("N",len(lbl))
    print("P",sum(lbl))



    if len(lbl)!= 0:
        print("ent_AUROC",roc_auc_score(lbl,ent_scores))
        print("DU_AUROC",roc_auc_score(lbl,du_scores))
        print("MI_AUROC ", roc_auc_score(lbl,mi_scores))
        print("ent_AUPR ", average_precision_score(lbl,ent_scores))
        print("DU_AUPR",average_precision_score(lbl,du_scores))
        print("MI_AUPR", average_precision_score(lbl,mi_scores))

    

    
    epoch_loss = running_loss # /1509
    # print("ged",np.mean(ged))
    
    return epoch_loss,np.mean(dice_coeff),segmentation[-1],y[-1],images,entropy[-1],data_uncertainty[-1],MI[-1],ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR
 
def compute_train_loss_and_train_with_OOD(train_loader, model, optimizer, use_gpu):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0

    dice_coeff = []

    for x,y,_ in train_loader:
        z = y.clone()
        # x,y,z = OOD_partial_mask2(x,y,z)
        # x,z = OOD_surround(x,z)
        # x,z = OOD_partial_mask1(x,z)
        # x,z = OOD_all_mask1(x,z)
        x,y,z = OOD_all_mask2(x,y,z)
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
        
        # forward pass
        outputs = model(x)

        batch_size = x.shape[0]
        mask = y[-1]
        images =x[-1]

        # compute loss 
        loss,prediction = model.loss(y)
        _, segg = torch.max(prediction, dim=1)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

        # track loss 
        running_loss += loss
        dice_ls = DSC_score_batch(batch_size,y,segg)
        dice_coeff.append(np.mean(dice_ls))
    
    epoch_loss = running_loss # /12078

    return epoch_loss,np.mean(dice_coeff),segg[-1],mask,images

def compute_eval_loss_with_OOD(test_loader, model, optimizer, use_gpu):
    """
    computes the evaluation epoch loss on the evaluation set
    """
    # model.eval()

    dice_coeff = []
    ent_score=[]
    du_score =[]
    mi_score =[]
    lbls = []
    ent_AUROC = 0
    DU_AUROC = 0
    MI_AUROC = 0
    ent_AUPR = 0
    DU_AUPR = 0
    MI_AUPR = 0

    Odu_scores = []
    Omutual_scores = []
    Oent_scores = []
    Olbl = []

    ent_AUROCs= []
    DU_AUROCs= []
    MI_AUROCs= []
    ent_AUPRs= []
    DU_AUPRs= []
    MI_AUPRs = []
    num = 0

    running_loss = 0.0
    with torch.no_grad():
        for x,y,_ in test_loader:
            z = y.clone()
            
            # Experiment 2
            # x,y,z = OOD_surround(x,y,z)
            
            # Experiment 3
            x,y,z = OOD_partial_mask2(x,y,z)

            # Experiment 4
            # x,y,z = OOD_all_mask2(x,y,z)
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
            
            # forward pass
            
            
            outputs = model(x)
            images = x[-1]

            # compute loss 
            N = 1114
            # N = 3
            loss,prediction= model.loss(y)
            # _, segmentation = torch.max(prediction, dim=1)
            segmentation,maxseg =torch.max(prediction, dim=1)  
            A = segmentation * maxseg
            segmentation = segmentation.float()
            segmentation[segmentation >0.9] = 0 
            segmentation = segmentation + A
            segmentation =  segmentation.expand((1,) + segmentation.shape)
            prediction = prediction.expand((1,) + prediction.shape)
            batch_size = x.shape[0]
            
            for i in range(N):
                outputs = model(x)
                loss,pred= model.loss(y)
                # _, segg = torch.max(pred, dim=1)
                segg,maxsegg =torch.max(pred, dim=1)  
                A = segg * maxsegg
                segg = segg.float()
                segg[segg >0.9] = 0 
                segg = segg + A
                pred = pred.expand((1,) + pred.shape)
                dice_ls = DSC_score_batch(batch_size,y,segg)
                dice_coeff.append(np.mean(dice_ls))
                segg = segg.expand((1,) + segg.shape)
                segmentation = torch.cat((segmentation,segg))
                prediction = torch.cat((prediction,pred))

            # track loss 
            running_loss += loss
            segmentation = segmentation.float()
            segmentation = segmentation.mean(dim=0) #[batch,128,128]

            # Uncertainty Estimation
            # prediction [1115,2,2,128,128]
            softmax = nn.Softmax(dim = 2)
            prediction = softmax(prediction)
            relu = nn.ReLU(inplace=False)
            prediction = relu(prediction)+1e-6
            
            prediction = prediction.permute(1,0,2,3,4) #[batch,N,2,128,128]
            # to make sure there is no infinity value at uncertainty estimation
            
            prob_tot= prediction.mean(dim=1) #[batch,2,128,128]
            
            
            entropy = -torch.sum(prob_tot*torch.log(prob_tot), dim=1) #[batch,128,128]
            

            data_uncertainty = -torch.sum(prediction * torch.log(prediction),dim=2) #[batch,N,128,128]
            data_uncertainty = data_uncertainty.mean(dim=1)#[batch,128,128]
            # print("du",data_uncertainty)
            MI = entropy-data_uncertainty #[batch,128,128]

            
            
            # AUROC,AUPR for classification

            mask = y[-1]
            entropy_score,dataU_score,MI_score,labels = misclassification_scores_MC(batch_size,entropy,data_uncertainty,MI,segmentation,y)
            OOD_entropy_score,OOD_dataU_score,OOD_MI_score,OOD_labels =OODdetection_scores_MC2(batch_size,entropy,data_uncertainty,MI,segmentation,y,z)



            lbl=list(labels.detach().cpu().numpy())
            du_scores= list(dataU_score.detach().cpu().numpy())
            ent_scores =list(entropy_score.detach().cpu().numpy())
            mi_scores = list(MI_score.detach().cpu().numpy())

            Olbl.append(list(OOD_labels.detach().cpu().numpy()))
            Odu_scores.append(list(OOD_dataU_score.detach().cpu().numpy()))
            Oent_scores.append(list(OOD_entropy_score.detach().cpu().numpy()))
            Omutual_scores.append(list(OOD_MI_score.detach().cpu().numpy())) 

            ent_score.append(ent_scores)
            du_score.append(du_scores)
            mi_score.append(mi_scores)
            lbls.append(lbl)
            if num == 8:
                visual_test(segmentation[-1],y[-1],dice_ls,x[-1],entropy[-1],data_uncertainty[-1],MI[-1],'17')
            num = num + 1
            # visual_test(segmentation[-1],mask,dice_ls,images,entropy[-1],data_uncertainty[-1],MI[-1])
            
    

    lbl = list(np.concatenate(lbls).flat)
    ent_scores =list(np.concatenate(ent_score).flat)
    du_scores =list(np.concatenate(du_score).flat)
    mi_scores =list(np.concatenate(mi_score).flat)

    Olbl = list(np.concatenate(Olbl).flat)
    Oent_scores =list(np.concatenate(Oent_scores).flat)
    Odu_scores =list(np.concatenate(Odu_scores).flat)
    Omutual_scores =list(np.concatenate(Omutual_scores).flat)

    print("MN",len(lbl))
    print("MP",sum(lbl))
    print("ON",len(Olbl))
    print("OP",sum(Olbl))



    if len(lbl)!= 0:


        ent_AUROC = roc_auc_score(lbl,ent_scores)
        DU_AUROC = roc_auc_score(lbl,du_scores)
        MI_AUROC = roc_auc_score(lbl,mi_scores)
        ent_AUPR=average_precision_score(lbl,ent_scores)
        DU_AUPR = average_precision_score(lbl,du_scores)
        MI_AUPR = average_precision_score(lbl,mi_scores)
        print("ent_AUROC",roc_auc_score(lbl,ent_scores))
        print("DU_AUROC",roc_auc_score(lbl,du_scores))
        print("MI_AUROC ", roc_auc_score(lbl,mi_scores))
        print("ent_AUPR ", average_precision_score(lbl,ent_scores))
        print("DU_AUPR",average_precision_score(lbl,du_scores))
        print("MI_AUPR", average_precision_score(lbl,mi_scores))

    
    if len(Olbl)!= 0:
        # misclassification_AU(lbl,ent_scores,du_scores,mutual_scores)
        
        print("Oent_AUROC",roc_auc_score(Olbl,Oent_scores))
        print("ODU_AUROC",roc_auc_score(Olbl,Odu_scores))
        print("OMI_AUROC ", roc_auc_score(Olbl,Omutual_scores))
        print("Oent_AUPR ", average_precision_score(Olbl,Oent_scores))
        print("ODU_AUPR",average_precision_score(Olbl,Odu_scores))
        print("OMI_AUPR", average_precision_score(Olbl,Omutual_scores))

    
    epoch_loss = running_loss # /1509
    
    
    return epoch_loss,np.mean(dice_coeff),segmentation[-1],mask,images,entropy[-1],data_uncertainty[-1],MI[-1],ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR
