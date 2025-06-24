import torch
import torch.nn as nn
from torch.nn import functional as F

from deepwonder.SR.SR_Unet import Unet4_no_end, CNN, DoubleConv, SingleConv
from deepwonder.SR.SR_Unet import UP_trans_block, UP_trans_resize_block
import numpy as np


########################################################################################################
########################################################################################################
class SR_Net(nn.Module):
    def __init__(self, net_type, in_ch, out_ch, f_num):
        super(SR_Net, self).__init__()
        print("net_type -----> ", net_type)
        if net_type == 'trans':
            self.net = SR_trans_Net(in_ch, out_ch, f_num).cuda()
        if net_type == 'trans_mini':
            self.net = SR_trans_Net1(in_ch, out_ch, f_num).cuda()
        if net_type == 'trans_mini2':
            self.net = SR_trans_Net2(in_ch, out_ch, f_num).cuda()
        if net_type == 'imresize':
            self.net = SR_imresize_Net(in_ch, out_ch, f_num).cuda()
        if net_type == 'fast1':
            self.net = SR_trans_Net_fast1(in_ch, out_ch, f_num).cuda()
        if net_type == 'ps':
            self.net = SR_ps_Net(in_ch, out_ch, f_num).cuda()


    def forward(self, x, up_rate):
        x_sr, x_sr_da = self.net(x, up_rate)
        return x_sr, x_sr_da 
    


########################################################################################################
########################################################################################################
class SR_imresize_Net(nn.Module):
    def __init__(self, in_ch, out_ch, f_num):
        super(SR_imresize_Net, self).__init__()

        self.net1 = CNN(in_ch, f_num, f_num)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num)
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
        # print('x2_ave -----> ',x2_ave.shape)
        for i in range(0, x2_p.shape[0]):
            x2_ave[i,:,:,:] = F.conv2d(x2_p[i,:,:,:].unsqueeze(1), weight_ave, padding=0)

        x3 = x2.clone().detach()
        x3[:,:,init_r::up_rate,init_c::up_rate] = x2_ave[:,:,init_r::up_rate,init_c::up_rate].clone().detach()
        # x2_ave = x2.clone().detach()
        # x3 = x2.clone().detach()
        return x2, x3 # , x2_ave


########################################################################################################
########################################################################################################
class SR_trans_Net(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, 
    k_size_net1=3, k_size_net2=3, k_size_up=3, k_size_deno=3):
        super(SR_trans_Net, self).__init__()
        # k_size_net1=3, k_size_net2=15, k_size_up=15, k_size_deno=15)

        self.net1 = CNN(in_ch, f_num, f_num, k_size = k_size_net1, bias=False)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num, k_size = k_size_net2, bias=False)
        self.up_block = UP_trans_resize_block(f_num, f_num, f_num, k_size = k_size_up, bias = False)
        self.net_deno = Unet4_no_end(out_ch, out_ch, 4, k_size = k_size_deno, bias=False)
        self.dp_layer = nn.Dropout(p=0.3)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def ave_merge(self, x2, up_rate):
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
        return x3, x2_ave

    def forward(self, x, up_rate):
        # print('x ---> ',np.max(x.cpu().detach().numpy()))
        x1 = self.net1(x)
        # print('x1 ---> ',np.max(x1.cpu().detach().numpy()))
        x1 = self.up_block(x1)
        out_h = int(x.shape[2]*up_rate)
        out_w = int(x.shape[3]*up_rate)

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)

        # x3, x2_ave = self.ave_merge(x2, up_rate)
        x3 = x2.clone() #.detach()
        x_masked = self.dp_layer(x3)
        x_masked_out = self.net_deno(x_masked)
        return x2, x_masked_out # x3 # , x2_ave




########################################################################################################
########################################################################################################
from .SR_Unet import CNN2, UP_upsample_block, UP_ps_resize_block1
from torch.cuda.amp import autocast

class SR_trans_Net_fast1(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, 
    k_size_net1=3, k_size_net2=3, k_size_up=3, k_size_deno=3, if_deno_net=0):
        super(SR_trans_Net_fast1, self).__init__()
        # k_size_net1=3, k_size_net2=15, k_size_up=15, k_size_deno=15)

        self.net1 = CNN(in_ch, f_num, f_num, k_size = k_size_net1, bias=False)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num, k_size = k_size_net2, bias=False)
        self.up_block = UP_upsample_block(f_num, f_num, f_num, k_size = k_size_up, bias = False)
        if if_deno_net:
            self.net_deno = CNN2(out_ch, out_ch, 4, k_size = k_size_deno, bias=False)
        self.dp_layer = nn.Dropout(p=0.3)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.if_deno_net = if_deno_net

    def ave_merge(self, x2, up_rate):
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
        return x3, x2_ave

    @autocast()
    def forward(self, x, up_rate):
        # print('x ---> ',np.max(x.cpu().detach().numpy()))
        x1 = self.net1(x)
        # print('x1 ---> ',np.max(x1.cpu().detach().numpy()))
        x1 = self.up_block(x1)
        out_h = int(x.shape[2]*up_rate)
        out_w = int(x.shape[3]*up_rate)

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)

        # x3, x2_ave = self.ave_merge(x2, up_rate)
        x3 = x2.clone() #.detach()
        x_masked = self.dp_layer(x3)
        if self.if_deno_net:
            x_masked_out = self.net_deno(x_masked)
        else: 
            x_masked_out = x2
        return x2, x_masked_out # x3 # , x2_ave



########################################################################################################
########################################################################################################
class SR_trans_Net1(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, 
    k_size_net1=3, k_size_net2=3, k_size_up=3, k_size_deno=3, if_deno_net=0):
        super(SR_trans_Net1, self).__init__()
        # k_size_net1=3, k_size_net2=15, k_size_up=15, k_size_deno=15)
        self.if_deno_net = if_deno_net

        self.net1 = CNN(in_ch, f_num, f_num, k_size = k_size_net1, bias=False, basic_module=SingleConv)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num, k_size = k_size_net2, bias=False, basic_module=SingleConv)
        self.up_block = UP_trans_resize_block(f_num, f_num, f_num, k_size = k_size_up, bias = False, basic_module=SingleConv)
        if if_deno_net:
            self.net_deno = Unet4_no_end(out_ch, out_ch, 4, k_size = k_size_deno, bias=False)
        self.dp_layer = nn.Dropout(p=0.3)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, up_rate):
        # print('x ---> ',np.max(x.cpu().detach().numpy()))
        x1 = self.net1(x)
        # print('x1 ---> ',np.max(x1.cpu().detach().numpy()))
        x1 = self.up_block(x1)
        out_h = int(x.shape[2]*up_rate)
        out_w = int(x.shape[3]*up_rate)

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)

        if self.if_deno_net:
            x3 = x2.clone() #.detach()
            x_masked = self.dp_layer(x3)
            x_masked_out = self.net_deno(x_masked)
        else: 
            x_masked_out = x2.clone()
        return x2, x_masked_out # x3 # , x2_ave



########################################################################################################
########################################################################################################
class SR_trans_Net2(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, 
    k_size_net1=3, k_size_net2=3, k_size_up=3, k_size_deno=3, if_deno_net=0):
        super(SR_trans_Net2, self).__init__()
        # k_size_net1=3, k_size_net2=15, k_size_up=15, k_size_deno=15)
        self.if_deno_net = if_deno_net

        self.net1 = CNN(in_ch, f_num, f_num, k_size = k_size_net1, bias=False, basic_module=SingleConv)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num, k_size = k_size_net2, bias=False, basic_module=SingleConv)
        self.up_block = UP_trans_resize_block(f_num, f_num, f_num, k_size = k_size_up, bias = False)
        if if_deno_net:
            self.net_deno = Unet4_no_end(out_ch, out_ch, 4, k_size = k_size_deno, bias=False)
        self.dp_layer = nn.Dropout(p=0.3)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, up_rate):
        # print('x ---> ',np.max(x.cpu().detach().numpy()))
        x1 = self.net1(x)
        # print('x1 ---> ',np.max(x1.cpu().detach().numpy()))
        x1 = self.up_block(x1)
        out_h = int(x.shape[2]*up_rate)
        out_w = int(x.shape[3]*up_rate)

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)

        if self.if_deno_net:
            x3 = x2.clone() #.detach()
            x_masked = self.dp_layer(x3)
            x_masked_out = self.net_deno(x_masked)
        else: 
            x_masked_out = x2.clone()
        return x2, x_masked_out # x3 # , x2_ave
    



########################################################################################################
########################################################################################################
class SR_ps_Net(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, 
    k_size_net1=3, k_size_net2=3, k_size_up=3, k_size_deno=3, if_deno_net=0):
        super(SR_ps_Net, self).__init__()
        # k_size_net1=3, k_size_net2=15, k_size_up=15, k_size_deno=15)
        self.if_deno_net = if_deno_net

        self.net1 = CNN(in_ch, f_num, f_num, k_size = k_size_net1, bias=False, basic_module=SingleConv)
        self.net2 = Unet4_no_end(f_num, out_ch, f_num, k_size = k_size_net2, bias=False, basic_module=SingleConv)
        self.up_block = UP_ps_resize_block1(f_num, f_num, f_num, k_size = k_size_up, bias = False)
        if if_deno_net:
            self.net_deno = Unet4_no_end(out_ch, out_ch, 4, k_size = k_size_deno, bias=False)
        self.dp_layer = nn.Dropout(p=0.3)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, up_rate):
        # print('x ---> ',np.max(x.cpu().detach().numpy()))
        x1 = self.net1(x)
        # print('x1 ---> ',np.max(x1.cpu().detach().numpy()))
        x1 = self.up_block(x1)
        out_h = int(x.shape[2]*up_rate)
        out_w = int(x.shape[3]*up_rate)

        x1_resize = F.interpolate(x1, (out_h, out_w),  mode='bilinear', align_corners=False)
        x2 = self.net2(x1_resize)

        if self.if_deno_net:
            x3 = x2.clone() #.detach()
            x_masked = self.dp_layer(x3)
            x_masked_out = self.net_deno(x_masked)
        else: 
            x_masked_out = x2.clone()
        return x2, x_masked_out # x3 # , x2_ave