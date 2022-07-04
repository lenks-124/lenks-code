import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class DoubleAttention(nn.Module):
    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)# 这里有可能进行降维操作，也可以直接设置为和下面一样
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h, w = x.shape
        A=self.convA(x) # b,c_m,h,w
        B=self.convB(x) # b,c_n,h,w
        V=self.convV(x) # b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1),dim=-1) # 在h x w这个维度上进行softmax。相当于对每个位置进行softmax。
        attention_vectors=F.softmax(V.view(b,self.c_n,-1),dim=1) # 相当于在每个点上 对所有通道进行softmax
        # 第一步：特征提取
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1))  # b, c_m, c_n
        # 第二步：特征分布

        # tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ = torch.bmm(global_descriptors,attention_vectors)

        tmpZ=tmpZ.view(b,self.c_m,h,w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)
        return tmpZ

if __name__ == '__main__':
    x=torch.randn(100,512,6,6)
    A2 = DoubleAttention(512,128,128,True)
    output=A2(x)
    print(output.shape)
    # torch.Size([100, 512, 6, 6])
