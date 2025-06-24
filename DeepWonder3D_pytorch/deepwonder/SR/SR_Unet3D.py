import torch.nn as nn
import torch
from torch import autograd
import numpy as np


########################################################################################
########################################################################################
class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size,if_nonlinear=True):
        super(DoubleConv3d, self).__init__()
        self.k_size_f = int(k_size/2)

        if if_nonlinear:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=False),
                nn.ReLU(inplace=True),
                # nn.ReLU(inplace=True),

                nn.Conv3d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=False),
                nn.ReLU(inplace=True),
                # nn.ReLU(inplace=True),
            )
        if not if_nonlinear:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=False),

                nn.Conv3d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=False),
            )

    def forward(self, input):
        return self.conv(input)


########################################################################################
########################################################################################
class SingleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size,if_nonlinear=True):
        super(SingleConv3d, self).__init__()
        self.k_size_f = int(k_size/2)

        if if_nonlinear:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=False),
                nn.ReLU(inplace=True),
                # nn.ReLU(inplace=True),
            )
        if not if_nonlinear:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=False),
            )

    def forward(self, input):
        return self.conv(input)


########################################################################################
########################################################################################
class Unet4_3d(nn.Module):
    def __init__(self, in_ch, out_ch, f_num):
        super(Unet4_3d, self).__init__()
        k_size = 3
        self.conv1 = DoubleConv3d(in_ch, f_num, k_size)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv3d(f_num, f_num, k_size)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv3d(f_num, f_num, k_size)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DoubleConv3d(f_num, f_num, k_size)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = DoubleConv3d(f_num, f_num, k_size)

        self.up6 = nn.Upsample(scale_factor=2, mode="nearest") #
        self.conv6 = DoubleConv3d(f_num*2, f_num, k_size)
        self.up7 = nn.Upsample(scale_factor=2, mode="nearest") # 
        self.conv7 = DoubleConv3d(f_num*2, f_num, k_size)
        self.up8 = nn.Upsample(scale_factor=2, mode="nearest") # 
        self.conv8 = DoubleConv3d(f_num*2, f_num, k_size)
        self.up9 = nn.Upsample(scale_factor=2, mode="nearest") #
        self.conv9 = DoubleConv3d(f_num*2, f_num, k_size)
        self.conv10 = DoubleConv3d(f_num, out_ch, k_size, if_nonlinear=False)

    def forward(self, x):

        x_h, x_w, x_t = x.size()[-3:]
        net_step = 16
        padding_h = int(np.ceil(x_h/net_step)*net_step-x_h)
        padding_w = int(np.ceil(x_w/net_step)*net_step-x_w)
        padding_t = int(np.ceil(x_t/net_step)*net_step-x_t)
        x = nn.ReplicationPad3d((0, padding_h, 0, padding_w, 0, padding_t))(x)

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
        c10 = self.conv10(c9)

        out = c10[..., :x_h, :x_w, :x_t]
        return out



class CNN3D(nn.Module):
    def __init__(self, in_ch, out_ch = 256, f_num = 64):
        super(CNN3D, self).__init__()
        k_size = 3
        self.conv1 = SingleConv3d(in_ch, f_num, k_size)
        self.conv2 = SingleConv3d(f_num, f_num, k_size)
        self.conv3 = SingleConv3d(f_num, out_ch, k_size)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return c3




class CNN3D_end(nn.Module):
    def __init__(self, in_ch, out_ch = 256, f_num = 64):
        super(CNN3D_end, self).__init__()
        k_size = 3
        self.conv1 = SingleConv3d(in_ch, f_num, k_size)
        self.conv2 = SingleConv3d(f_num, f_num, k_size)
        self.conv3 = SingleConv3d(f_num, out_ch, k_size, if_nonlinear=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        return c3

'''        
x_h, x_w = x.size()[-2:]
net_step = 16
padding_h = int(np.ceil(x_h/net_step)*net_step-x_h)
padding_w = int(np.ceil(x_w/net_step)*net_step-x_w)
x = nn.ReplicationPad2d((0, padding_h, 0, padding_w))(x)
'''