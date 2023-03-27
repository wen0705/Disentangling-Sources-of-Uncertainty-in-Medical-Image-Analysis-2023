# Author: Yutong Wen
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as KL
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as td
from DisU.distributions import ReshapedDistribution
import math


def RKL_Loss(prediction, mask, rkl_mask):
    batch_size = prediction.size()[0]
    # set loss function
    rev = lambda x : Dirichlet(x)

    # uncertainty estimation
    rkl_mask = rkl_mask.long() # [5, 128, 128]
    mask = mask.long()

    # target alpha
    target_alpha = F.one_hot(mask,num_classes=2) #[5,128,128,2]
    target_alpha = target_alpha*10 + 1 #scaling
    
    #Out-of-distribution convertion
    mask_clone = rkl_mask.clone() # [5, 128, 128]
    mask_clone[mask_clone!=5]=0
    for i in range(batch_size):
        indices = mask_clone[i].nonzero() 
        # print("indices",indices.size())# size(900,2) 
        if len(indices)!= 0 :
            target_alpha[i][indices[:,0],indices[:,1]] = 1
            
    
    # pred alpha
    pred_alpha = torch.sigmoid(prediction) #[5,2,128,128]
    pred_alpha = pred_alpha * 10 + 1
    # pred_alpha = pred_alpha * 2 + 1
    pred_alpha =  pred_alpha.permute(0,2,3,1) #[5,128,128,2]
    loss = torch.mean(KL(rev(pred_alpha), rev(target_alpha)))
    return loss,pred_alpha

def CE_Loss(mask,prediction):
    x = mask.shape[-1]
    y = mask.shape[-2]
    CEloss = nn.CrossEntropyLoss()
    relu = nn.ReLU(inplace=False)
    return CEloss(
        relu(prediction),
        mask.view(-1, x, y).long()
    )

def SSN_Loss(prediction,mean,cov_diag,cov_factor,num_classes,batch_size,pixel_size,rank,target,epsilon):
    # mean of the distribution
        logit_mean = mean
        # _, segmentation = torch.max(mean, dim=1)
        logit_mean = logit_mean.view((batch_size, -1)) #[5,32768]

        #convariance diagnoal
        cov_diag = cov_diag.exp() #[5,2,128,128]
        cov_diag = cov_diag.view((batch_size, -1)) 
   
        # covariance factor
        cov_factor = cov_factor #[5,20,128,128]
        cov_factor = cov_factor.view((batch_size, rank, num_classes, -1)) #[5,10,2,16384]
        cov_factor = cov_factor.flatten(2, 3) #[5,10,32768]
        cov_factor = cov_factor.transpose(1, 2) #[5,32768,10]

        # set 0 outside the ROI
        # mask = mask.unsqueeze(1).expand((batch_size,num_classes) + mask.shape[1:]).reshape(batch_size,-1)

        # cov_diag = cov_diag * mask
        cov_diag = cov_diag  +  epsilon #[5,32768]
        # cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_factor = cov_factor 
        
        try:
            base_distribution = td.LowRankMultivariateNormal(loc=logit_mean, cov_factor=cov_factor, cov_diag=cov_diag)
        except:
            print("covariance became not pd")
            base_distribution = td.Independent(td.Normal(loc=logit_mean, scale= torch.sqrt(cov_diag)), 1)

        event_shape = (num_classes,) + prediction.shape[2:]
        distribution = ReshapedDistribution(base_distribution, event_shape)

        # logits = logit_mean.reshape((batch_size, num_classes, pixel_size,-1))
        
   
        M = 20
        # reparameterize
        samples = distribution.rsample((M // 2,)) 
        mean = distribution.mean.unsqueeze(0)
        samples = samples - mean
        logit_samples = torch.cat([samples, -samples]) + mean #[20, 5, 2, 128, 128]



        flatsize = M * batch_size
        target = target.expand((M,) + target.shape) #[M,5,1,128,128]
        logit_samples = logit_samples.view((flatsize,num_classes,-1)) #[5M,2,16384]
        target = target.reshape((flatsize, -1)) #[5M,16384]
        target = target.long()
        # lg = nn.Softmax(dim=1)
        # logit_samples =lg(logit_samples)


        # for uncertainty estimation
        # relu = nn.ReLU(inplace=False)
        # logit_samples = relu(logit_samples) + 1e-6
      

        log_prob = -F.cross_entropy(logit_samples, target, reduction='none').view((M, batch_size, -1)) #[1000, 5, 16384] # this is log
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(M))
        loss = -loglikelihood
        return loss, logit_samples


