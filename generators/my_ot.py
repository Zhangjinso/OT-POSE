from lib2to3.pgen2 import token
from telnetlib import X3PAD
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def sinkhorn(feature1, feature2, epsilon=1e-8, gamma=1, max_iter=300, image=None):
    """
    Sinkhorn algorithm
    Parameters
    ----------
    feature2 : torch.Tensor  # source image feature with source pose
        Feature for points cloud 1. Used to computed transport cost. 
        Size B x W X H x C.
    feature1 : torch.Tensor # target pose feature
        Feature for points cloud 2. Used to computed transport cost. 
        Size B x W X H x C.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.
    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    support = 1

    h, w = feature1.shape[2:]
    
    feature1 = rearrange(feature1, 'b c h w -> b (h w) c')
    feature2 = rearrange(feature2, 'b c h w -> b (h w) c')
    if image is not None:
        image = F.interpolate(image, (h, w)).detach()
        image = rearrange(image, 'b c h w -> b (h w) c')
    #print(feature1.shape, feature2.shape)

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / 0.03) * support
    # print(K.shape)

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    T = T/ torch.sum(T, dim=2, keepdim=True)
    # print(T.max())

    warped = torch.matmul(T, feature2)
    warped = rearrange(warped, 'b (h w) c -> b c h w', h=h, w=w)
    if image is not None:
        warped_image = torch.matmul(T, image)
        warped_image = rearrange(warped_image, 'b (h w) c -> b c h w', h=h, w=w)
        return warped, warped_image
    else:
        return warped


if __name__ == "__main__":
    a = torch.ones([2, 256, 16,11]).cuda()
    b = torch.ones([2, 256, 16,11]).cuda()
    v = sinkhorn(a,b)
    print(v.shape)
    
    # import cv2
    # import numpy as np
    # p = '/home/zjs/2022/dataset/fashion_high/test/'
    # ra = cv2.imread(p+'fashionMENJackets_Vestsid0000724701_1front.jpg')
    # rb = cv2.imread(p+'fashionMENJackets_Vestsid0000724701_2side.jpg')
    # ra = cv2.resize(ra, (100,100))
    # rb = cv2.resize(rb, (100, 100))
    # a = torch.from_numpy(ra).permute(2,1, 0).unsqueeze(0).cuda()
    # b = torch.from_numpy(rb).permute(2,1,0).unsqueeze(0).cuda()
    # print(a.shape)
    # T, warped = log_otp_solver(a, b)
    # print(T)
    # warped = np.array(warped.squeeze(0).permute(1,2,0).cpu())
    # t = np.concatenate([ra,rb,warped], 1)
    # cv2.imwrite('test.png', t)
