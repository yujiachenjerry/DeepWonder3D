import torch
import torch.nn as nn
from torch.nn import functional as F

from deepwonder.SR.Unet import Unet4_no_end, CNN
from deepwonder.SR.Unet3D import Unet4_3d, CNN3D
import numpy as np

class bg_sr_net(nn.Module):
    def __init__(self, in_ch, out_ch, f_num):
        super(bg_sr_net, self).__init__()

        self.net1 = Unet4_no_end(in_ch, f_num, f_num)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num)
        # self.net3 = Unet4_no_end(out_ch, out_ch, f_num)
        self.in_ch = in_ch
        self.out_ch = out_ch


    def forward(self, x, up_rate):
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
        for i in range(0, x2_p.shape[1]):
            x2_ave[:,i,:,:] = F.conv2d(x2_p[:,i,:,:].unsqueeze(1), weight_ave, padding=0)

        x3 = x2.clone().detach()
        x3[:,:,init_r::up_rate,init_c::up_rate] = x2_ave[:,:,init_r::up_rate,init_c::up_rate].clone().detach()

        return x2, x3, x2_ave


########################################################################################################

class SR_Net_3D(nn.Module):
    def __init__(self, in_ch, out_ch, f_num):
        super(SR_Net_3D, self).__init__()

        self.net1 = CNN3D(in_ch, f_num, f_num)
        self.net2 = Unet4_3d(f_num, out_ch, f_num)
        # self.net3 = Unet4_no_end(out_ch, out_ch, f_num)
        self.in_ch = in_ch
        self.out_ch = out_ch


    def forward(self, x, up_rate):
        # print('x ---> ',x.shape)
        x1 = self.net1(x)
        # print('x1 ---> ',x1.shape)
        x1_view = x1.view(x1.shape[0], x1.shape[1]*x1.shape[2], x1.shape[3], x1.shape[4])
        # print('x1_view ---> ',x1_view.shape)

        out_h = x.shape[-2]*up_rate
        out_w = x.shape[-1]*up_rate

        x1_view_resize = F.interpolate(x1_view, (out_h, out_w),  mode='bilinear', align_corners=False)
        # print('x1_view_resize ---> ',x1_view_resize.shape)
        # print('x1_view_resize ---> ',x1_view_resize.shape[0], x1.shape[1], x1.shape[2], x1_view_resize.shape[-2], x1_view_resize.shape[-1])
        x1_resize = x1_view_resize.view(x1_view_resize.shape[0], x1.shape[1], x1.shape[2], x1_view_resize.shape[-2], x1_view_resize.shape[-1])
        # print('x1_resize ---> ',x1_resize.shape)

        x2 = self.net2(x1_resize)
        # print('x2 ---> ',x2.shape)
        x2_clone = x2.clone().detach()
        x2 = x2.view(x2.shape[0]*x2.shape[-3], 1, x2.shape[-2], x2.shape[-1])
        ###########################################################################
        init_r = int((up_rate-1)//2)
        init_c = int((up_rate-1)//2)
        # print('init_r -----> ',init_r,' up_rate -----> ',up_rate)
        k_ave = [[1/8,1/8,1/8],
                 [1/8,   0,1/8],
                 [1/8,1/8,1/8]]
        k_ave = torch.FloatTensor(k_ave).unsqueeze(0).unsqueeze(0)

        weight_ave = nn.Parameter(data=k_ave,requires_grad=False).cuda()
        # print('x2 ---> ',x2.shape)
        x2_p = nn.ReplicationPad2d((1, 1, 1, 1))(x2)
        x2_ave = x2.clone().detach()
        # print('x2_ave -----> ',x2_ave.shape)
        for i in range(0, x2_p.shape[0]):
            x2_ave[i,:,:,:] = F.conv2d(x2_p[i,:,:,:].unsqueeze(1), weight_ave, padding=0)
        x2_ave = x2_ave.view(x2_clone.shape[0], x2_clone.shape[-3], x2_clone.shape[-2], x2_clone.shape[-1])
        
        x3 = x2.clone().detach()
        x3 = x3.view(x2_clone.shape[0], x2_clone.shape[-3], x2_clone.shape[-2], x2_clone.shape[-1])
        # print('x3 ---> ',x3.shape)
        # print('x2_ave ---> ',x2_ave.shape)
        x3[:,:,init_r::up_rate,init_c::up_rate] = x2_ave[:,:,init_r::up_rate,init_c::up_rate].clone().detach()

        # print('x2 ---> ',x2.shape)
        # print('x3 ---> ',x3.shape)
        # print('x2_ave ---> ',x2_ave.shape)
        x2_clone = x2_clone.view(x2_clone.shape[0], 1, x2_clone.shape[-3], x2_clone.shape[-2], x2_clone.shape[-1])
        x3 = x3.view(x3.shape[0], 1, x3.shape[-3], x3.shape[-2], x3.shape[-1])
        x2_ave = x2_ave.view(x2_ave.shape[0], 1, x2_ave.shape[-3], x2_ave.shape[-2], x2_ave.shape[-1])
        return x2_clone, x3, x2_ave