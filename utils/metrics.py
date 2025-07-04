import os 
import pickle
import numpy as np
from tqdm import tqdm
from scipy import linalg
from multiprocessing import Pool

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value

def iou_calc(pred, true, void, class_num):
    ious = np.zeros(class_num)
    smooth = 1e-6

    # 디버깅을 위해 pred와 true의 크기를 출력합니다.
    print(f'Debug: pred shape: {pred.shape}, true shape: {true.shape}')
    
    pred_1d = pred.reshape(-1)
    true_1d = true.reshape(-1)
    
    print(f'Debug: pred_1d shape: {pred_1d.shape}, true_1d shape: {true_1d.shape}')
    
    pred_1d_count = np.bincount(pred_1d, minlength=class_num)
    true_1d_count = np.bincount(true_1d, minlength=class_num)
    
    print(f'Debug: pred_1d_count: {pred_1d_count}, true_1d_count: {true_1d_count}')
    
    categorical_array = (class_num * true_1d) + pred_1d
    
    confusion_mat_1d = np.bincount(categorical_array, minlength=class_num*class_num)
    confusion_mat = confusion_mat_1d.reshape((class_num, class_num))
    
    print(f'Debug: confusion_mat shape: {confusion_mat.shape}')
    
    intersection = np.diag(confusion_mat)
    union = true_1d_count + pred_1d_count - intersection
    union = union - confusion_mat[11, :]
    
    print(f'Debug: intersection: {intersection}, union: {union}')
    
    for i in range(len(true_1d_count)):
        if true_1d_count[i] == 0:
            ious[i] = np.nan
        else:
            ious[i] = (intersection[i] + smooth) / (union[i] + smooth)
            
    if void == True:
        ious[11] = np.nan

    return ious

