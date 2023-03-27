# Author: Yutong Wen
import numpy as np
import torch
from DisU.evaluations import DSC_score_batch,misclassification_scores,OODdetection_scores2,OODdetection_scores
from model.OOD_generator import OOD_surround,OOD_partial_mask2,OOD_partial_mask1,OOD_all_mask1,OOD_all_mask2,OOD_real
from sklearn.metrics import roc_auc_score,average_precision_score
from DisU.visualization import visual_test
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl

def compute_eval_loss_without_OOD(test_loader, model, optimizer, use_gpu):
    """
    computes for the entire test set
    """
    model.eval()

    dice_coeff = []
    du_scores = []
    mutual_scores = []
    ent_scores = []
    entropy = []
    lbl = []
    ged =[]
    num = 0

    running_loss = 0.0
    with torch.no_grad():
        # for x,y,_ in test_loader:
        for x_ori,y,_ in test_loader:
            
        #     b = x_ori.shape[0]
            xs = torch.cat((x_ori, x_ori, x_ori,x_ori), 1)
            x = xs.reshape(4,1,128,128)
            y = y.reshape(4,128,128)
            z = y.clone()
            
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
            
            
            
            # forward pass
            outputs= model(x)
            # print("xx-------xx-------xx")

            # compute loss 
            loss, pred_alpha= model.loss(y,z)
            batch_size = x.shape[0]
            pred_alpha0 = torch.sum(pred_alpha,-1,keepdim=True)
            prob_mu = pred_alpha/pred_alpha0 #([5, 128, 128, 2])
            prob_mu = prob_mu.permute(0,3,1,2)#([5, 2,128, 128])
            print("xxx",prob_mu.shape)
    

            # performance DSC  
            _, segmentation = torch.max(prob_mu, dim=1)   
            dice_ls = DSC_score_batch(batch_size,y,segmentation)
        

            # AUROC, AUPR Scores
            entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation= misclassification_scores(batch_size,pred_alpha,y,segmentation)
    

            dice_coeff.append(np.mean(dice_ls))
            images = x[-1]
            mask = y[-1]
            lbl.append(list(labels.detach().cpu().numpy()))
            du_scores.append(list(dataU_score.detach().cpu().numpy()))
            ent_scores.append(list(entropy_score.detach().cpu().numpy()))
            mutual_scores.append(list(MI_score.detach().cpu().numpy()))

            

            running_loss += loss
            num = num + 1
            
            #         visual_test(segmentation[i],y[i],dice_ls,x[i],entropy[i],dataUncertainty[i],mutualInformation[i],print)
                


    lbl = list(np.concatenate(lbl).flat)
    ent_scores =list(np.concatenate(ent_scores).flat)
    du_scores =list(np.concatenate(du_scores).flat)
    mutual_scores =list(np.concatenate(mutual_scores).flat)
    # print("ged",np.mean(ged))

    

    ent_AUROC = 0
    DU_AUROC = 0
    MI_AUROC = 0
    ent_AUPR = 0
    DU_AUPR = 0
    MI_AUPR = 0
    print('mN',len(lbl))
    print('mP',sum(lbl))
    if len(lbl)!= 0:
        # misclassification_AU(lbl,ent_scores,du_scores,mutual_scores)

        print("ent_AUROC",roc_auc_score(lbl,ent_scores))
        print("DU_AUROC",roc_auc_score(lbl,du_scores))
        print("MI_AUROC ", roc_auc_score(lbl,mutual_scores))
        print("ent_AUPR ", average_precision_score(lbl,ent_scores))
        print("DU_AUPR",average_precision_score(lbl,du_scores))
        print("MI_AUPR", average_precision_score(lbl,mutual_scores))
    

    
    epoch_loss = running_loss # /1509

    # return epoch_loss
    
    return epoch_loss,dice_coeff,mask,entropy,dataUncertainty,mutualInformation,segmentation[-1],images,ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR

def compute_train_loss_and_train_without_OOD(train_loader, model, optimizer, use_gpu):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0

    dice_coeff = []
    # ged =[]
    
    for x,y,_ in train_loader:

        z = y.clone()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
        
        # forward pass
        outputs = model(x)
        # compute loss 
        loss, pred_alpha= model.loss(y,z)

        # pred mu and Cross entropy loss
        batch_size = x.shape[0]
        
        pred_alpha0 = torch.sum(pred_alpha,-1,keepdim=True)
        prob_mu = pred_alpha/pred_alpha0 #([5, 128, 128, 2])
        prob_mu = prob_mu.permute(0,3,1,2)#([5, 2,128, 128])
   

        # performance DSC  
        _, segmentation = torch.max(prob_mu, dim=1)   
        dice_ls = DSC_score_batch(batch_size,y,segmentation)
        
        dice_coeff.append(np.mean(dice_ls))
        images = x[-1]
        mask = y[-1]

        # backward pass
        # with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
            # track loss 
        running_loss += loss
    
    epoch_loss = running_loss # /12078
    ged = []
   
    
    return epoch_loss,dice_coeff,mask,segmentation[-1],images,ged

def compute_eval_loss_with_OOD(test_loader, model, optimizer, use_gpu):
    """
    computes for the entire test set
    """
    model.eval()

    dice_coeff = []
    du_scores = []
    mutual_scores = []
    ent_scores = []
    entropy = []
    lbl = []
    Odu_scores = []
    Omutual_scores = []
    Oent_scores = []
    Oentropy = []
    Olbl = []
    Odu_scores2 = []
    Omutual_scores2 = []
    Oent_scores2 = []
    Oentropy = []
    Olbl2 = []
    num = 0

    running_loss = 0.0
    with torch.no_grad():
        for x,y,_ in test_loader:
            # y = y.reshape((4,128,128))
            
            z = y.clone() 
            x,y,z = OOD_real(x,y,z)
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
            
            # Experiment 2
            # x,y,z = OOD_surround(x,y,z)

            # Experiment 3
            # x,y,z = OOD_partial_mask2(x,y,z)

            # Experiment 4
            # x,y,z = OOD_all_mask2(x,y,z)
            
            # forward pass
            outputs= model(x)

            # compute loss 
            loss, pred_alpha= model.loss(y,z)
            batch_size = x.shape[0]
            pred_alpha0 = torch.sum(pred_alpha,-1,keepdim=True)
            prob_mu = pred_alpha/pred_alpha0 #([5, 128, 128, 2])
            prob_mu = prob_mu.permute(0,3,1,2)#([5, 2,128, 128])
 
            # soft-segmentation
            # similarity
            Q = 1 - torch.abs(prob_mu[:,1,:,:] - prob_mu[:,0,:,:]) / torch.sum(prob_mu,dim=1)
            Q = prob_mu[:,1,:,:]/torch.sqrt(prob_mu[:,0,:,:] * prob_mu[:,1,:,:])
            Q = Q.float()
            Q[Q<0.5] = 0
            Q[Q>=0.1]=0.6
            segmentation,maxseg =torch.max(prob_mu, dim=1)  
            A = segmentation * maxseg
            segmentation = segmentation.float()
            segmentation[segmentation >0.9] = 0 
            segmentation = segmentation + A
            # _, segmentation = torch.max(prob_mu, dim=1)   
            # s_shape = segmentation.shape
            # Q = Q.reshape(s_shape)
            # segmentation = segmentation + Q
            dice_ls = DSC_score_batch(batch_size,y,segmentation)
            

            # AUROC, AUPR Scores
            entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation= misclassification_scores(batch_size,pred_alpha,y,segmentation)
            OOD_entropy_score2,OOD_dataU_score2,OOD_MI_score2,OOD_labels2,_,_,_ =OODdetection_scores2(batch_size,pred_alpha,z,y,segmentation)
            OOD_entropy_score,OOD_dataU_score,OOD_MI_score,OOD_labels,_,_,_ =OODdetection_scores(batch_size,pred_alpha,z,y,segmentation)

        
            dice_coeff.append(np.mean(dice_ls))
            images = x[-1]
            mask = y[-1]
            lbl.append(list(labels.detach().cpu().numpy()))
            du_scores.append(list(dataU_score.detach().cpu().numpy()))
            ent_scores.append(list(entropy_score.detach().cpu().numpy()))
            mutual_scores.append(list(MI_score.detach().cpu().numpy()))

            Olbl.append(list(OOD_labels.detach().cpu().numpy()))
            Odu_scores.append(list(OOD_dataU_score.detach().cpu().numpy()))
            Oent_scores.append(list(OOD_entropy_score.detach().cpu().numpy()))
            Omutual_scores.append(list(OOD_MI_score.detach().cpu().numpy()))

            Olbl2.append(list(OOD_labels2.detach().cpu().numpy()))
            Odu_scores2.append(list(OOD_dataU_score2.detach().cpu().numpy()))
            Oent_scores2.append(list(OOD_entropy_score2.detach().cpu().numpy()))
            Omutual_scores2.append(list(OOD_MI_score2.detach().cpu().numpy()))

            running_loss += loss
            
            OOD = test_loader.__getitem__[494][0].float()
            OOD[OOD<0.5] = 0.0
            OOD[-1,:,:30,:] = 0.0
            OOD[-1,:,90:,:] = 0.0
            OOD[-1,:,:,50:] = 0.0

            x = x + OOD
            x = x.float()
            x[x>1] = 1
                # visual_test(segmentation[-1],y[-1],dice_ls,x[-1],entropy[-1],dataUncertainty[-1],mutualInformation[-1],False)
            num = num + 1

    lbl = list(np.concatenate(lbl).flat)
    ent_scores =list(np.concatenate(ent_scores).flat)
    du_scores =list(np.concatenate(du_scores).flat)
    mutual_scores =list(np.concatenate(mutual_scores).flat)

    Olbl = list(np.concatenate(Olbl).flat)
    Oent_scores =list(np.concatenate(Oent_scores).flat)
    Odu_scores =list(np.concatenate(Odu_scores).flat)
    Omutual_scores =list(np.concatenate(Omutual_scores).flat)

    Olbl2 = list(np.concatenate(Olbl2).flat)
    Oent_scores2 =list(np.concatenate(Oent_scores2).flat)
    Odu_scores2 =list(np.concatenate(Odu_scores2).flat)
    Omutual_scores2 =list(np.concatenate(Omutual_scores2).flat)
    print('mN',len(lbl))
    print('mP',sum(lbl))
    print('oN',len(Olbl))
    print('oP',sum(Olbl))
    print('oN2',len(Olbl2))
    print('oP2',sum(Olbl2))

    ent_AUROC = 0
    DU_AUROC = 0
    MI_AUROC = 0
    ent_AUPR = 0
    DU_AUPR = 0
    MI_AUPR = 0
    if len(lbl)!= 0:
        # misclassification_AU(lbl,ent_scores,du_scores,mutual_scores)
        
        ent_AUROC = roc_auc_score(lbl,ent_scores)
        DU_AUROC = roc_auc_score(lbl,du_scores)
        MI_AUROC = roc_auc_score(lbl,mutual_scores)
        ent_AUPR = average_precision_score(lbl,ent_scores)
        DU_AUPR = average_precision_score(lbl,du_scores)
        MI_AUPR = average_precision_score(lbl,mutual_scores)
        print("ent_AUROC",roc_auc_score(lbl,ent_scores))
        print("DU_AUROC",roc_auc_score(lbl,du_scores))
        print("MI_AUROC ", roc_auc_score(lbl,mutual_scores))
        print("ent_AUPR ", average_precision_score(lbl,ent_scores))
        print("DU_AUPR",average_precision_score(lbl,du_scores))
        print("MI_AUPR", average_precision_score(lbl,mutual_scores))
    
    if len(Olbl)!= 0:
        # misclassification_AU(lbl,ent_scores,du_scores,mutual_scores)
        
        print("Oent_AUROC",roc_auc_score(Olbl,Oent_scores))
        print("ODU_AUROC",roc_auc_score(Olbl,Odu_scores))
        print("OMI_AUROC ", roc_auc_score(Olbl,Omutual_scores))
        print("Oent_AUPR ", average_precision_score(Olbl,Oent_scores))
        print("ODU_AUPR",average_precision_score(Olbl,Odu_scores))
        print("OMI_AUPR", average_precision_score(Olbl,Omutual_scores))

        print("Oent_AUROC2",roc_auc_score(Olbl2,Oent_scores2))
        print("ODU_AUROC2",roc_auc_score(Olbl2,Odu_scores2))
        print("OMI_AUROC2 ", roc_auc_score(Olbl2,Omutual_scores2))
        print("Oent_AUPR2 ", average_precision_score(Olbl2,Oent_scores2))
        print("ODU_AUPR2",average_precision_score(Olbl2,Odu_scores2))
        print("OMI_AUPR2", average_precision_score(Olbl2,Omutual_scores2))

    
    epoch_loss = running_loss # /1509

    
    
    return epoch_loss,dice_coeff,mask,entropy,dataUncertainty,mutualInformation,segmentation[-1],images,ent_AUROC,DU_AUROC,MI_AUROC,ent_AUPR,DU_AUPR,MI_AUPR

def compute_train_loss_and_train_with_OOD(train_loader, model, optimizer, use_gpu):
    """
    computes the losses for every batch; so basically the epoch loss
    """
    model.train()

    running_loss = 0.0

    dice_coeff = []
    ged =[]
    
    for x,y,_ in train_loader:
        z = y.clone()
        x,y,z = OOD_real(x,y,z)

        
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
        
        # forward pass
        
        # x,y,z = OOD_partial_mask2(x,y,z)
        # x,z = OOD_partial_mask1(x,z)
        # x,z = OOD_all_mask1(x,z)
        # x,y,z = OOD_all_mask2(x,y,z)
        # x,z, = OOD_surround(x,z)
        outputs = model(x)
        # compute loss 
        loss, pred_alpha= model.loss(y,z)

        # pred mu and Cross entropy loss
        batch_size = x.shape[0]
        
        pred_alpha0 = torch.sum(pred_alpha,-1,keepdim=True)
        prob_mu = pred_alpha/pred_alpha0 #([5, 128, 128, 2])
        prob_mu = prob_mu.permute(0,3,1,2)#([5, 2,128, 128])
   

        # performance DSC  
        _, segmentation = torch.max(prob_mu, dim=1)   
        dice_ls = DSC_score_batch(batch_size,y,segmentation)
        dice_coeff.append(np.mean(dice_ls))
        images = x[-1]
        mask = y[-1]

        # backward pass
        # with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
            # track loss 
        running_loss += loss
    
    epoch_loss = running_loss # /12078
   
    
    return epoch_loss,dice_coeff,mask,segmentation[-1],images,ged