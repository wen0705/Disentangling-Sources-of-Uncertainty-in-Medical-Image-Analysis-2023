# Author: Yutong Wen
import torch
from sklearn.metrics import jaccard_score
import numpy as np


# DSC
def DSC_score(target,pred):
    if (1 not in target.long()) and (1 in pred.long()):
        dice = torch.Tensor([0])

    elif (1 not in target.long()) and (1 not in pred.long()):
        dice = torch.Tensor([1])

    else:
        pred_f = pred.flatten()
        target_f = target.flatten().cuda()
        intersection = torch.sum(pred_f * target_f)
        dice = 2 * intersection/ (torch.sum(pred_f) + torch.sum(target_f))
    return dice

def DSC_score_batch(batch_size,mask,segmentation):
    dice_ls = []
    for i in range(batch_size):
        # print("value",DSC_score(mask[i],segmentation[i]))
        dice_ls.append(DSC_score(mask[i],segmentation[i]).item())
    
    return dice_ls

def distance_func(pred,target):
    #d(x,y) = 1 -IoU(x,y)
    if (1 not in target) and (1 in pred):
        IoU = torch.Tensor([0])
    elif (1 not in target) and (1 not in pred.long()):
        IoU = torch.Tensor([1])
    else:
        IoU = jaccard_score(pred, target,average="micro")
    return 1 - IoU

def generalised_energy_distance(pred,target):
    pred = pred.long()
    target = target.long()

    #change gpu to cpu
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    d_yy = []
    d_sy = []
    d_ss = []
    for i in range(4):
        d_sy.append(distance_func(pred[i],target[i]).item())
        for j in range(4):
            d_yy.append(distance_func(target[i],target[j]).item())
            d_ss.append(distance_func(pred[i],pred[j]).item())
    
    return 2*np.mean(d_sy) -np.mean(d_ss)-np.mean(d_yy)

def generalised_energy_distance_MCD(pred,target,num_samples):
    pred = pred.float()
    # pred = pred.reshape((4,num_samples,128,128))
    seg = pred.mean(dim = 0)
    pred = pred.long()
    target = target.long()
    seg = seg.long()
    


    #change gpu to cpu
    seg = seg.detach().cpu()
    target = target.detach().cpu()
    
    # seg = pred.clone()
    
    d_yy = []
    d_ss =[]
    d_sy = []
    for i in range(4):
        d_sy.append(distance_func(seg[i],target[i]).item())
        for j in range(4):
            d_yy.append(distance_func(target[i],target[j]).item())
            d_ss.append(distance_func(seg[i],seg[j]).item())
    
        # for p in range(num_samples):
        #     #pred [M,b,128,128]
        #     pred_first = pred[0]
        #     pred_first = pred_first.reshape((1,4,128,128))
        #     pred_last = pred[1:]
        #     pred = torch.cat((pred_last,pred_first))
            

            # for q in range(num_samples):
            # print("xx",pred[:,i].shape)
            # print("---",seg[:,i].shape)
            # d_ss.append(distance_func(pred[:,i].flatten(),seg[:,i].flatten()).item())
            # d_sy.append(distance_func(seg[p,i],target[i]).item())
    return 2*np.mean(d_sy) -np.mean(d_ss)-np.mean(d_yy)

    

# def generalised_energy_distance_unet_batch(batch_size,mask,segmentation):
#     ls = []
#     for i in range(batch_size):
  
#        ls.append(generalised_energy_distance_unet(segmentation[i],mask[i]).item())
#     return ls
# UNCERTAINTY ESTIMATION
def uncertainty_estimation(alphas):
    alpha0 = torch.sum(alphas, dim=-1, keepdim=True) #Size([128, 128, 1]) 
    probs = alphas / alpha0 #[128,128,2]

    entropy = -torch.sum(probs*torch.log(probs), dim=-1)
    data_uncertainty = -torch.sum((alphas/alpha0)*(torch.digamma(alphas+1)-torch.digamma(alpha0+1)), dim=-1)
    MI = entropy -data_uncertainty
    return entropy,data_uncertainty,MI

def misclassification_scores(batch_size,pred_alpha,rkl_mask,segmentation):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    ent = []
    du = []
    mi = []
    for i in range(batch_size):
        entropy,dataUncertainty,mutualInformation= uncertainty_estimation(pred_alpha[i])
        ent.append(entropy)
        du.append(dataUncertainty)
        mi.append(mutualInformation)
        
        diff = rkl_mask[i].flatten()-segmentation[i].flatten()
        
        diff = diff.long()
        entropy_t = entropy.flatten()
        entropy_score = torch.cat((entropy_score,entropy_t))
        dataU_score = torch.cat((dataU_score,dataUncertainty.flatten()))
        MI_score = torch.cat((MI_score,mutualInformation.flatten()))
        label = torch.abs(diff)
        label = label.float()
        labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels,ent,du,mi
    # return entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation

def OODdetection_scores(batch_size,pred_alpha,rkl_mask,mask,segmentation):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        entropy,dataUncertainty,mutualInformation= uncertainty_estimation(pred_alpha[i])
        if 5 in rkl_mask[i].long():
            diff = rkl_mask[i].flatten()-segmentation[i].flatten()
            
            diff = diff.long()
            entropy_t = entropy.flatten()[diff != 0 ]
            entropy_score = torch.cat((entropy_score,entropy_t))
            dataU_score = torch.cat((dataU_score,dataUncertainty.flatten()[diff != 0 ]))
            MI_score = torch.cat((MI_score,mutualInformation.flatten()[diff != 0 ]))
            label = torch.abs(diff[diff != 0 ])
            label[label < 2] = 0
            label[label > 2] = 1
            label = label.float()
            labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation

# OOD positive example, rest of them negative
def OODdetection_scores2(batch_size,pred_alpha,rkl_mask,mask,segmentation):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = rkl_mask.clone()
    labels = labels.long()
    labels[labels<4] = 0
    labels[labels== 5] = 1
    labels = labels.flatten()

    for i in range(batch_size):
        entropy,dataUncertainty,mutualInformation= uncertainty_estimation(pred_alpha[i])
        entropy_t = entropy.flatten()
        entropy_score = torch.cat((entropy_score,entropy_t))
        dataU_score = torch.cat((dataU_score,dataUncertainty.flatten()))
        MI_score = torch.cat((MI_score,mutualInformation.flatten()))
    return entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation

def OODdetection_scores_MC2(batch_size,entropy,dataUncertainty,mutualInformation,segmentation,mask,rkl_mask):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = rkl_mask.clone()
    labels = labels.long()
    labels[labels<4] = 0
    labels[labels== 5] = 1
    labels = labels.flatten()
    entropy_score = entropy.flatten()
    dataU_score = dataUncertainty.flatten()
    MI_score = mutualInformation.flatten()
    return entropy_score,dataU_score,MI_score,labels

def OODdetection_scores_MC(batch_size,entropy,dataUncertainty,mutualInformation,segmentation,mask,rkl_mask):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        if 5 in rkl_mask[i].long():
            diff = rkl_mask[i].flatten()-segmentation[i].flatten()
            diff = diff.long()
            entropy_t = entropy[i].flatten()[diff != 0 ]
            entropy_score = torch.cat((entropy_score,entropy_t))
            dataU_score = torch.cat((dataU_score,dataUncertainty[i].flatten()[diff != 0 ]))
            MI_score = torch.cat((MI_score,mutualInformation[i].flatten()[diff != 0 ]))
            label = torch.abs(diff[diff != 0 ])
            label[label < 2] = 0
            label[label > 2] = 1
            label = label.float()
            labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels

def OODdetection_scores_MC3(batch_size,entropy,dataUncertainty,mutualInformation,segmentation,mask,rkl_mask):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        if 2 in rkl_mask[i].long():
            diff = rkl_mask[i].flatten()-segmentation[i].flatten()
            diff = diff.long()
            diff[rkl_mask[i].flatten() == 2] = 5
            entropy_t = entropy[i].flatten()[diff != 0 ]
            entropy_score = torch.cat((entropy_score,entropy_t))
            dataU_score = torch.cat((dataU_score,dataUncertainty[i].flatten()[diff != 0 ]))
            MI_score = torch.cat((MI_score,mutualInformation[i].flatten()[diff != 0 ]))
            label = torch.abs(diff[diff != 0 ])
            label[label < 2] = 0
            label[label > 2] = 1
            label = label.float()
            labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels

def misclassification_scores_MC(batch_size,entropy,dataUncertainty,mutualInformation,segmentation,mask):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        diff = mask[i].flatten()-segmentation[i].flatten()
        
        diff = diff.long()
        entropy_t = entropy[i].flatten()
        entropy_score = torch.cat((entropy_score,entropy_t))
        dataU_score = torch.cat((dataU_score,dataUncertainty[i].flatten()))
        MI_score = torch.cat((MI_score,mutualInformation[i].flatten()))
        label = torch.abs(diff)
        label = label.float()
        labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels

def misclassification_scores_MC3(batch_size,entropy,dataUncertainty,mutualInformation,segmentation,mask):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        diff = mask[i].flatten()-segmentation[i].flatten()
        
        diff = diff.long()
        entropy_t = entropy[i].flatten()
        entropy_score = torch.cat((entropy_score,entropy_t))
        dataU_score = torch.cat((dataU_score,dataUncertainty[i].flatten()))
        MI_score = torch.cat((MI_score,mutualInformation[i].flatten()))
        label = torch.abs(diff)
        label = label.float()
        label[label>1] = 1
        labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels


def scores(batch_size,pred_alpha,rkl_mask,segmentation):
    entropy_score = torch.tensor([]).cuda()
    dataU_score = torch.tensor([]).cuda()
    MI_score = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    for i in range(batch_size):
        entropy,dataUncertainty,mutualInformation= uncertainty_estimation(pred_alpha[i])
        if 5 in rkl_mask[i]:
        # change the OOD place to (1,1)
            
            diff = rkl_mask[i].flatten()-segmentation[i].flatten()
            
            diff = diff.long()
            entropy_t = entropy.flatten()
            entropy_t = entropy_t[diff != 0 ]
            
            entropy_score = torch.cat((entropy_score,entropy_t))
            dataU_score = torch.cat((dataU_score,dataUncertainty.flatten()[diff != 0 ]))
            MI_score = torch.cat((MI_score,mutualInformation.flatten()[diff != 0 ]))
            label = diff[diff != 0]
            label = torch.abs(label)
        
            label[label < 2] = 0
            label[label > 2] = 1
            label = label.float()
            labels = torch.cat((labels,label))
    return entropy_score,dataU_score,MI_score,labels,entropy,dataUncertainty,mutualInformation

