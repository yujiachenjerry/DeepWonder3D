import torch
import torch.nn as nn
from torch.nn import functional as F

from deepwonder.SR.Unet import Unet4_no_end, CNN
from deepwonder.SR.Unet3D import DoubleConv3d, CNN3D_end
import numpy as np


#########################################################################
#########################################################################
class SRST_Net(nn.Module):
    def __init__(self, in_ch, out_ch, f_num):
        super(SRST_Net, self).__init__()
        self.net1 = CNN(in_ch, f_num, f_num)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num)
        self.end_net = CNN3D_end(out_ch, out_ch, f_num//4, )
        # self.net3 = Unet4_no_end(out_ch, out_ch, f_num)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, up_rate, stage):
        # print('x ---> ',x.shape)
        x = x.view(x.shape[-3], x.shape[-4], x.shape[-2], x.shape[-1])
        x1 = self.net1(x)

        out_h = x.shape[2]*up_rate
        out_w = x.shape[3]*up_rate

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)
        ###########################################################################
        init_r = int((up_rate-1)//2)
        init_c = int((up_rate-1)//2)
        # print('init_r -----> ',init_r,' up_rate -----> ',up_rate)
        k_ave = [[1/8,1/8,1/8],
                 [1/8,   0,1/8],
                 [1/8,1/8,1/8]]
        k_ave = torch.FloatTensor(k_ave).unsqueeze(0).unsqueeze(0)

        weight_ave = nn.Parameter(data=k_ave,requires_grad=False).cuda()
        x2_p = nn.ReplicationPad2d((1, 1, 1, 1))(x2)
        x2_ave = x2.clone().detach()
        # print('x2_ave -----> ',x2_ave.shape)
        for i in range(0, x2_p.shape[0]):
            x2_ave[i,:,:,:] = F.conv2d(x2_p[i,:,:,:].unsqueeze(1), weight_ave, padding=0)

        x3 = x2.clone().detach()
        x3[:,:,init_r::up_rate,init_c::up_rate] = x2_ave[:,:,init_r::up_rate,init_c::up_rate].clone().detach()
        ###########################################################################
        # print('x2_ave -----> ',x2_ave.shape)
        # print('x3 -----> ',x3.shape)
        # print('x2 -----> ',x2.shape)
        
        x3 = x3.view(1, x3.shape[-3], x3.shape[-4], x3.shape[-2], x3.shape[-1])
        if stage==1:
            x4 = x3.clone().detach()
        if stage==2:
            x4 = self.end_net(x3)
        x2 = x2.view(1, x2.shape[-3], x2.shape[-4], x2.shape[-2], x2.shape[-1])        
        # print('x3 111 -----> ',x3.shape)
        # print('x2 111 -----> ',x2.shape)
        return x2, x3, x4