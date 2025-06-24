import torch.nn as nn
import torch
from torch import autograd
import numpy as np
from torch.nn import functional as F


###########################################################################################
###########################################################################################
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, bias=False ,if_nonlinear=True, if_residual=True):
        super(SingleConv, self).__init__()
        self.k_size_f = int(k_size/2)
        self.if_residual = if_residual

        if if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv2d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )
        if not if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv2d( in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )

    def forward(self, input):
        x1 = self.conv1(input)
        return x1



###########################################################################################
###########################################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, bias=False ,if_nonlinear=True, if_residual=True):
        super(DoubleConv, self).__init__()
        self.k_size_f = int(k_size/2)
        self.if_residual = if_residual

        if if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv2d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential( nn.Conv2d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )
        if not if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv2d( in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )
            self.conv2 = nn.Sequential( nn.Conv2d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )

    def forward(self, input):
        x1 = self.conv1(input)
        if self.if_residual:
            x2 = self.conv2(x1)+x1
        if not self.if_residual:
            x2 = self.conv2(x1)
        return x2


###########################################################################################
###########################################################################################
class Unet4_no_end(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, k_size = 15, bias=False, basic_module=DoubleConv):
        super(Unet4_no_end, self).__init__()
        self.up_type = 'upsample'  # or upsample trans

        self.conv1 = basic_module(in_ch, f_num, k_size, bias)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = basic_module(f_num, f_num, k_size, bias)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = basic_module(f_num, f_num, k_size, bias)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = basic_module(f_num, f_num, k_size, bias)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = basic_module(f_num, f_num, k_size, bias)

        if self.up_type=='upsample':
            self.up6 = nn.Upsample(scale_factor=2, mode="nearest")
            self.up7 = nn.Upsample(scale_factor=2, mode="nearest")
            self.up8 = nn.Upsample(scale_factor=2, mode="nearest") 
            self.up9 = nn.Upsample(scale_factor=2, mode="nearest")
        if self.up_type=='trans':
            self.up6 = nn.ConvTranspose2d(f_num, f_num, 2, stride=2)
            self.up7 = nn.ConvTranspose2d(f_num, f_num, 2, stride=2)
            self.up8 = nn.ConvTranspose2d(f_num, f_num, 2, stride=2)
            self.up9 = nn.ConvTranspose2d(f_num, f_num, 2, stride=2)

        self.conv6 = basic_module(f_num*2, f_num,  k_size, bias)
        self.conv7 = basic_module(f_num*2, f_num,  k_size, bias)
        self.conv8 = basic_module(f_num*2, f_num,  k_size, bias)
        self.conv9 = basic_module(f_num*2, out_ch, k_size, bias, if_nonlinear=False)

    def forward(self, x):
        x_h, x_w = x.size()[-2:]
        net_step = 16
        padding_h = int(np.ceil(x_h/net_step)*net_step-x_h)
        padding_w = int(np.ceil(x_w/net_step)*net_step-x_w)
        x = nn.ReplicationPad2d((0, padding_h, 0, padding_w))(x)

        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        out = c9[..., :x_h, :x_w]
        return out


###########################################################################################
###########################################################################################
class CNN(nn.Module):
    def __init__(self, in_ch, out_ch = 256, f_num = 64, k_size = 3, bias=False, basic_module=DoubleConv):
        super(CNN, self).__init__()
        self.conv1 = basic_module(in_ch, f_num, k_size, bias)
        self.conv2 = basic_module(f_num, f_num, k_size, bias)
        self.conv3 = basic_module(f_num, out_ch, k_size, bias)

    def forward(self, x):

        x_h, x_w = x.size()[-2:]
        net_step = 16
        padding_h = int(np.ceil(x_h/net_step)*net_step-x_h)
        padding_w = int(np.ceil(x_w/net_step)*net_step-x_w)
        x = nn.ReplicationPad2d((0, padding_h, 0, padding_w))(x)

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        out = c3[..., :x_h, :x_w]
        return out



###########################################################################################
###########################################################################################
class CNN2(nn.Module):
    def __init__(self, in_ch, out_ch = 256, f_num = 64, k_size = 3, bias=False):
        super(CNN2, self).__init__()
        self.conv1 = DoubleConv(in_ch, f_num, k_size, bias)
        self.conv3 = DoubleConv(f_num, out_ch, k_size, bias)

    def forward(self, x):

        x_h, x_w = x.size()[-2:]
        net_step = 16
        padding_h = int(np.ceil(x_h/net_step)*net_step-x_h)
        padding_w = int(np.ceil(x_w/net_step)*net_step-x_w)
        x = nn.ReplicationPad2d((0, padding_h, 0, padding_w))(x)

        c1 = self.conv1(x)
        c3 = self.conv3(c1)

        out = c3[..., :x_h, :x_w]
        return out



########################################################################################################
########################################################################################################
class UP_trans_block(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, k_size = 7):
        super(UP_trans_block, self).__init__()
        trans_k = 4
        assert trans_k%2 == 0
        trans_p = (trans_k-2)//2

        k_size_f = int(k_size/2)
        if_bias = True
        self.up1 = nn.ConvTranspose2d(in_ch, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c1 = DoubleConv(f_num, f_num, k_size, bias=if_bias)
        self.up2 = nn.ConvTranspose2d(f_num, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c2 = DoubleConv(f_num, f_num, k_size, bias=if_bias)
        self.up3 = nn.ConvTranspose2d(f_num, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c3 = DoubleConv(f_num, f_num, k_size, bias=if_bias)
        self.up4 = nn.ConvTranspose2d(f_num, out_ch, kernel_size=trans_k, stride=2, padding=trans_p, bias=False)

    def forward(self, x):
        x = self.up1(x)
        x = self.c1(x)
        x = self.up2(x)
        x = self.c2(x)
        x = self.up3(x)
        x = self.c3(x)
        x = self.up4(x)
        return x
    

########################################################################################################
########################################################################################################
class UP_trans_resize_block(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, k_size = 7, bias = False, basic_module=DoubleConv):
        super(UP_trans_resize_block, self).__init__()
        trans_k = 4
        assert trans_k%2 == 0
        trans_p = (trans_k-2)//2

        k_size_f = int(k_size/2)
        if_bias = bias
        self.up1 = nn.ConvTranspose2d(in_ch, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c1 = basic_module(f_num+f_num, f_num, k_size, bias=if_bias)
        self.up2 = nn.ConvTranspose2d(f_num, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c2 = basic_module(f_num+f_num, f_num, k_size, bias=if_bias)
        self.up3 = nn.ConvTranspose2d(f_num, f_num, kernel_size=trans_k, stride=2, padding=trans_p, bias=if_bias)
        self.c3 = basic_module(f_num+f_num, f_num, k_size, bias=if_bias)
        self.up4 = nn.ConvTranspose2d(f_num, out_ch, kernel_size=trans_k, stride=2, padding=trans_p, bias=False)
        self.c4 = basic_module(f_num+f_num, f_num, k_size, bias=if_bias)

    def forward(self, x):
        if_output_f = False

        x1_0 = x.clone()
        x1_up = self.up1(x1_0)
        # print('x1_up ---> ',np.max(x1_up.cpu().detach().numpy()))
        out_h1 = x1_up.shape[2]
        out_w1 = x1_up.shape[3]
        x1_0_inter = F.interpolate(x1_0, (out_h1, out_w1),  mode='bilinear', align_corners=False)
        # print('x1 -----> ',x1.shape)
        # print('x -----> ',x.shape)
        x1_cat = torch.cat([x1_0_inter, x1_up], dim=1)
        x1_out = self.c1(x1_cat)# +x1_up
        xf1 = x1_out.clone()

        x2_up = self.up2(x1_out)
        # print('x2_up ---> ',np.max(x2_up.cpu().detach().numpy()))
        out_h2 = x2_up.shape[2]
        out_w2 = x2_up.shape[3]
        x2_0_inter = F.interpolate(x1_0, (out_h2, out_w2),  mode='bilinear', align_corners=False)
        x2_cat = torch.cat([x2_0_inter, x2_up], dim=1)
        x2_out = self.c2(x2_cat)# +x2_up
        xf2 = x2_out.clone()

        x3_up = self.up3(x2_out)
        # print('x3_up ---> ',np.max(x3_up.cpu().detach().numpy()))
        out_h3 = x3_up.shape[2]
        out_w3 = x3_up.shape[3]
        x3_0_inter = F.interpolate(x1_0, (out_h3, out_w3),  mode='bilinear', align_corners=False)
        x3_cat = torch.cat([x3_0_inter, x3_up], dim=1)
        x3_out = self.c3(x3_cat)# +x3_up
        xf3 = x3_out.clone()

        x4_up = self.up4(x3_out)
        # print('x4_up ---> ',np.max(x4_up.cpu().detach().numpy()))
        out_h4 = x4_up.shape[2]
        out_w4 = x4_up.shape[3]
        x4_0_inter = F.interpolate(x1_0, (out_h4, out_w4),  mode='bilinear', align_corners=False)
        x4_cat = torch.cat([x4_0_inter, x4_up], dim=1)
        x4_out = self.c4(x4_cat)# +x4_up
        # xf4 = x.clone()
        if if_output_f:
            return x4_out, xf1, xf2, xf3
        if not if_output_f:
            return x4_out





########################################################################################################
########################################################################################################
class UP_upsample_block(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, k_size = 7, bias = False):
        super(UP_upsample_block, self).__init__()
        trans_k = 4
        assert trans_k%2 == 0
        trans_p = (trans_k-2)//2

        k_size_f = int(k_size/2)
        if_bias = bias
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.c1 = DoubleConv(f_num, f_num, k_size, bias=if_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.c2 = DoubleConv(f_num+f_num, f_num, k_size, bias=if_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.c3 = DoubleConv(f_num+f_num, f_num, k_size, bias=if_bias)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.c4 = DoubleConv(f_num+f_num, f_num, k_size, bias=if_bias)

    def forward(self, x):
        if_output_f = False

        x1_0 = x.clone()
        x1_up = self.up1(x1_0)
        x1_out = self.c1(x1_up)# +x1_up
        xf1 = x1_out.clone()

        x2_up = self.up2(x1_out)
        # print('x2_up ---> ',np.max(x2_up.cpu().detach().numpy()))
        out_h2 = x2_up.shape[2]
        out_w2 = x2_up.shape[3]
        x2_0_inter = F.interpolate(x1_0, (out_h2, out_w2),  mode='bilinear', align_corners=False)
        x2_cat = torch.cat([x2_0_inter, x2_up], dim=1)
        x2_out = self.c2(x2_cat)# +x2_up
        xf2 = x2_out.clone()

        x3_up = self.up3(x2_out)
        # print('x3_up ---> ',np.max(x3_up.cpu().detach().numpy()))
        out_h3 = x3_up.shape[2]
        out_w3 = x3_up.shape[3]
        x3_0_inter = F.interpolate(x2_out, (out_h3, out_w3),  mode='bilinear', align_corners=False)
        x3_cat = torch.cat([x3_0_inter, x3_up], dim=1)
        x3_out = self.c3(x3_cat)# +x3_up
        xf3 = x3_out.clone()

        x4_up = self.up4(x3_out)
        # print('x4_up ---> ',np.max(x4_up.cpu().detach().numpy()))
        out_h4 = x4_up.shape[2]
        out_w4 = x4_up.shape[3]
        x4_0_inter = F.interpolate(x3_out, (out_h4, out_w4),  mode='bilinear', align_corners=False)
        x4_cat = torch.cat([x4_0_inter, x4_up], dim=1)
        x4_out = self.c4(x4_cat)# +x4_up
        # xf4 = x.clone()
        if if_output_f:
            return x4_out, xf1, xf2, xf3
        if not if_output_f:
            return x4_out



########################################################################################################
########################################################################################################
class UP_ps_resize_block1(nn.Module):
    def __init__(self, in_ch, out_ch, f_num, k_size = 7, bias = False, basic_module=DoubleConv):
        super(UP_ps_resize_block1, self).__init__()
        pixel_shuffle_layer = nn.PixelShuffle(2)

        trans_k = 4
        assert trans_k%2 == 0
        trans_p = (trans_k-2)//2

        k_size_f = int(k_size/2)
        if_bias = bias
        self.up1 = pixel_shuffle_layer
        self.c1 = basic_module(in_ch//4, f_num, k_size, bias=if_bias)
        self.up2 = pixel_shuffle_layer
        self.c2 = basic_module(f_num//4, f_num, k_size, bias=if_bias)
        self.up3 = pixel_shuffle_layer
        self.c3 = basic_module(f_num//4, f_num, k_size, bias=if_bias)
        self.up4 = pixel_shuffle_layer
        self.c4 = basic_module(f_num//4, f_num, k_size, bias=if_bias)

    def forward(self, x):
        if_output_f = False

        x1_0 = x.clone()
        x1_up = self.up1(x1_0)
        # print('x1_up ---> ',np.max(x1_up.cpu().detach().numpy()))
        out_h1 = x1_up.shape[2]
        out_w1 = x1_up.shape[3]
        # x1_0_inter = F.interpolate(x1_0, (out_h1, out_w1),  mode='bilinear', align_corners=False)
        # print('x1 -----> ',x1.shape)
        # print('x -----> ',x.shape)
        # x1_cat = torch.cat([x1_0_inter, x1_up], dim=1)
        x1_out = self.c1(x1_up)# +x1_up
        xf1 = x1_out.clone()

        x2_up = self.up2(x1_out)
        # print('x2_up ---> ',np.max(x2_up.cpu().detach().numpy()))
        out_h2 = x2_up.shape[2]
        out_w2 = x2_up.shape[3]
        # x2_0_inter = F.interpolate(x1_0, (out_h2, out_w2),  mode='bilinear', align_corners=False)
        # x2_cat = torch.cat([x2_0_inter, x2_up], dim=1)
        x2_out = self.c2(x2_up)# +x2_up
        xf2 = x2_out.clone()

        x3_up = self.up3(x2_out)
        # print('x3_up ---> ',np.max(x3_up.cpu().detach().numpy()))
        out_h3 = x3_up.shape[2]
        out_w3 = x3_up.shape[3]
        # x3_0_inter = F.interpolate(x1_0, (out_h3, out_w3),  mode='bilinear', align_corners=False)
        # x3_cat = torch.cat([x3_0_inter, x3_up], dim=1)
        x3_out = self.c3(x3_up)# +x3_up
        xf3 = x3_out.clone()

        x4_up = self.up4(x3_out)
        # print('x4_up ---> ',np.max(x4_up.cpu().detach().numpy()))
        out_h4 = x4_up.shape[2]
        out_w4 = x4_up.shape[3]
        # x4_0_inter = F.interpolate(x1_0, (out_h4, out_w4),  mode='bilinear', align_corners=False)
        # x4_cat = torch.cat([x4_0_inter, x4_up], dim=1)
        x4_out = self.c4(x4_up)# +x4_up
        # xf4 = x.clone()
        if if_output_f:
            return x4_out, xf1, xf2, xf3
        if not if_output_f:
            return x4_out

