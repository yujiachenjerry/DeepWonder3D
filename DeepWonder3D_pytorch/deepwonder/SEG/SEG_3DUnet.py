import importlib

import torch
import torch.nn as nn

from .SEG_buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from .SEG_buildingblocks import TS_Encoder, TS_Decoder, DoubleConvTS
from .SEG_utils import create_feature_maps



###########################################################################################
###########################################################################################
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, bias=False ,if_nonlinear=True, if_residual=True):
        super(DoubleConv3D, self).__init__()
        self.k_size_f = int(k_size/2)
        self.if_residual = if_residual

        if if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential( nn.Conv3d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )
        if not if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv3d( in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )
            self.conv2 = nn.Sequential( nn.Conv3d(out_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )

    def forward(self, input):
        x1 = self.conv1(input)
        if self.if_residual:
            x2 = self.conv2(x1)+x1
        if not self.if_residual:
            x2 = self.conv2(x1)
        return x2


###########################################################################################
###########################################################################################
class SingleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, bias=False ,if_nonlinear=True, if_residual=True):
        super(SingleConv3D, self).__init__()
        self.k_size_f = int(k_size/2)
        self.if_residual = if_residual

        if if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv3d(in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias),
                                        # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.ReLU(inplace=True), )

        if not if_nonlinear:
            self.conv1 = nn.Sequential( nn.Conv3d( in_ch, out_ch, k_size, padding=self.k_size_f, bias=bias), )

    def forward(self, input):
        x1 = self.conv1(input)
        return x1


###########################################################################################
###########################################################################################
class Unet4_single_no_end_3D(nn.Module):
    def __init__(self, in_ch, out_ch, frame_num=64, f_num=4, k_size = 3, bias=False):
        super(Unet4_single_no_end_3D, self).__init__()
        self.up_type = 'upsample'  # or upsample trans
        self.final_pooling = nn.MaxPool3d((frame_num//16,1,1), stride = (frame_num//16,1,1))

        self.pool_222 = nn.MaxPool3d(2)
        self.pool_211 = nn.MaxPool3d((2,1,1))

        self.conv1 = SingleConv3D(in_ch, f_num, k_size, bias)
        self.conv2 = SingleConv3D(f_num, f_num, k_size, bias)
        self.conv3 = SingleConv3D(f_num, f_num, k_size, bias)
        self.conv4 = SingleConv3D(f_num, f_num, k_size, bias)
        self.conv5 = SingleConv3D(f_num, f_num, k_size, bias)

        self.up_222 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_122 = nn.Upsample(scale_factor=(1,2,2), mode="nearest")

        self.conv6 = SingleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv7 = SingleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv8 = SingleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv9 = SingleConv3D(f_num*2, out_ch, k_size, bias, if_nonlinear=False)

        self.final_activation = nn.Sigmoid()

        self.final_conv = nn.Conv2d(frame_num, 1, kernel_size=3, padding=1)
        self.dp_layer = nn.Dropout(p=0.2)
        # self.final_pooling = nn.Conv3d(out_channels, out_channels,  (128,1,1), padding=0)
        # self.final_pooling = nn.MaxPool3d((frame_num,1,1), stride = (frame_num,1,1))

    def forward(self, x):
        # print('x ---> ',x.shape)
        c1 = self.conv1(x)
        p1 = self.pool_222(c1)
        c2 = self.conv2(p1)
        p2 = self.pool_222(c2)
        c3 = self.conv3(p2)
        p3 = self.pool_222(c3)
        c4 = self.conv4(p3)
        p4 = self.pool_222(c4)
        c5 = self.conv5(p4)

        # print('c5 ---> ',c5.shape)
        up_6 = self.up_222(c5)
        # c4_p = self.pool_211(c4)
        # print('up_6 ---> ',up_6.shape,'c4 ---> ',c4.shape,)
        merge6 = torch.cat([up_6, c4], dim=1)
        # print('merge6 ---> ',merge6.shape)
        c6 = self.conv6(merge6)

        up_7 = self.up_222(c6)
        # c3_p = self.pool_211(self.pool_211(c3))
        # print('up_7 ---> ',up_7.shape,'c3_p ---> ',c3_p.shape,'c3 ---> ',c3.shape,)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up_222(c7)
        # c2_p = self.pool_211(self.pool_211(self.pool_211(c2)))
        # print('up_8 ---> ',up_8.shape,'c2_p ---> ',c2_p.shape,'c2 ---> ',c2.shape,)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up_222(c8)
        # c1_p = self.pool_211(self.pool_211(self.pool_211(self.pool_211(c1))))
        # print('up_9 ---> ',up_9.shape,'c1_p ---> ',c1_p.shape,'c1 ---> ',c1.shape,)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # print('c9 ---> ',c9.shape)
        c9_squeeze = c9.squeeze(dim=1)
        # out = self.final_pooling(c9)
        out_squeeze = self.final_conv(c9_squeeze)
        out = out_squeeze.unsqueeze(dim=1)
        # print('out ---> ',out.shape)
        # out = self.final_activation(out)
        return out



###########################################################################################
###########################################################################################
class Unet4_no_end_3D(nn.Module):
    def __init__(self, in_ch, out_ch, frame_num=64, f_num=4, k_size = 3, bias=False):
        super(Unet4_no_end_3D, self).__init__()
        self.up_type = 'upsample'  # or upsample trans
        self.final_pooling = nn.MaxPool3d((frame_num//16,1,1), stride = (frame_num//16,1,1))

        self.pool_222 = nn.MaxPool3d(2)
        self.pool_211 = nn.MaxPool3d((2,1,1))

        self.conv1 = DoubleConv3D(in_ch, f_num, k_size, bias)
        self.conv2 = DoubleConv3D(f_num, f_num, k_size, bias)
        self.conv3 = DoubleConv3D(f_num, f_num, k_size, bias)
        self.conv4 = DoubleConv3D(f_num, f_num, k_size, bias)
        self.conv5 = DoubleConv3D(f_num, f_num, k_size, bias)

        self.up_222 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_122 = nn.Upsample(scale_factor=(1,2,2), mode="nearest")

        self.conv6 = DoubleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv7 = DoubleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv8 = DoubleConv3D(f_num*2, f_num,  k_size, bias)
        self.conv9 = DoubleConv3D(f_num*2, out_ch, k_size, bias, if_nonlinear=False)

        self.final_activation = nn.Sigmoid()

        self.final_conv = nn.Conv2d(frame_num, 1, kernel_size=3, padding=1)
        self.dp_layer = nn.Dropout(p=0.2)
        # self.final_pooling = nn.Conv3d(out_channels, out_channels,  (128,1,1), padding=0)
        # self.final_pooling = nn.MaxPool3d((frame_num,1,1), stride = (frame_num,1,1))

    def forward(self, x):
        if_use_dp = 1
        # print('x ---> ',x.shape)
        c1 = self.conv1(x)
        if if_use_dp:
            c1 = self.dp_layer(c1)
        p1 = self.pool_222(c1)
        c2 = self.conv2(p1)
        if if_use_dp:
            c2 = self.dp_layer(c2)
        p2 = self.pool_222(c2)
        c3 = self.conv3(p2)
        if if_use_dp:
            c3 = self.dp_layer(c3)
        p3 = self.pool_222(c3)
        c4 = self.conv4(p3)
        if if_use_dp:
            c4 = self.dp_layer(c4)
        p4 = self.pool_222(c4)
        c5 = self.conv5(p4)
        if if_use_dp:
            c5 = self.dp_layer(c5)

        # print('c5 ---> ',c5.shape)
        up_6 = self.up_222(c5)
        # c4_p = self.pool_211(c4)
        # print('up_6 ---> ',up_6.shape,'c4 ---> ',c4.shape,)
        merge6 = torch.cat([up_6, c4], dim=1)
        # print('merge6 ---> ',merge6.shape)
        c6 = self.conv6(merge6)

        up_7 = self.up_222(c6)
        # c3_p = self.pool_211(self.pool_211(c3))
        # print('up_7 ---> ',up_7.shape,'c3_p ---> ',c3_p.shape,'c3 ---> ',c3.shape,)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up_222(c7)
        # c2_p = self.pool_211(self.pool_211(self.pool_211(c2)))
        # print('up_8 ---> ',up_8.shape,'c2_p ---> ',c2_p.shape,'c2 ---> ',c2.shape,)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up_222(c8)
        # c1_p = self.pool_211(self.pool_211(self.pool_211(self.pool_211(c1))))
        # print('up_9 ---> ',up_9.shape,'c1_p ---> ',c1_p.shape,'c1 ---> ',c1.shape,)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # print('c9 ---> ',c9.shape)
        c9_squeeze = c9.squeeze(dim=1)
        # out = self.final_pooling(c9)
        out_squeeze = self.final_conv(c9_squeeze)
        out = out_squeeze.unsqueeze(dim=1)
        # print('out ---> ',out.shape)
        # out = self.final_activation(out)
        return out



class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, frame_num, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        # self.final_pooling = nn.Conv3d(out_channels, out_channels,  (128,1,1), padding=0)
        self.final_pooling = nn.MaxPool3d((frame_num,1,1), stride = (frame_num,1,1))
        self.final_activation = nn.Sigmoid()
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        # final_pooling = nn.MaxPool3d((x.shape[2],1,1), stride = (x.shape[2],1,1))
        # x = final_pooling(x)
        # print(x.shape)
        x = self.final_pooling(x)
        x = self.final_activation(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x



class T_conv_net(nn.Module):
    def __init__(self, in_channels, frame_num, tc_f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(T_conv_net, self).__init__()

        self.end_frame_num = round(frame_num/8)
        tc_k_size = (3,1,1)
        sc_k_size = (1,3,3)
        padding_size_t = (1,0,0)
        padding_size_s = (0,1,1)
        pool_kernel_size = (2,1,1)
        '''
        self.layer1 = DoubleConv(in_channels, tc_f_maps, encoder=False,
                                kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.layer2 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.pooling1 = nn.MaxPool3d(kernel_size=pool_kernel_size)
        

        self.layer3 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.layer4 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.pooling2 = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.layer5 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.layer6 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.pooling3 = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.layer7 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.layer8 = DoubleConv(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size=tc_k_size, order=layer_order, num_groups=num_groups, padding=padding_size)
        self.final_pooling = nn.MaxPool3d((self.end_frame_num,1,1), stride = (self.end_frame_num,1,1))
        '''
        # self.final_pooling = nn.Conv3d(tc_f_maps, tc_f_maps,  kernel_size=(self.end_frame_num,1,1), stride = (self.end_frame_num,1,1), padding=0)
        self.layer1 = DoubleConvTS(in_channels, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.layer2 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.pooling1 = nn.MaxPool3d(kernel_size=pool_kernel_size)
        

        self.layer3 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.layer4 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.pooling2 = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.layer5 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.layer6 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.pooling3 = nn.MaxPool3d(kernel_size=pool_kernel_size)

        self.layer7 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.layer8 = DoubleConvTS(tc_f_maps, tc_f_maps, encoder=False,
                        kernel_size_T=tc_k_size, kernel_size_S=sc_k_size, order=layer_order, 
                        num_groups=num_groups, padding_T=padding_size_t, padding_S=padding_size_s)
        self.final_pooling = nn.MaxPool3d((self.end_frame_num,1,1), stride = (self.end_frame_num,1,1))
    

    def forward(self, x):
        x = self.layer1(x)
        # print('layer1 -----> ',x.shape)
        x = self.layer2(x)
        x = self.pooling1(x)
        # print('layer2 -----> ',x.shape)
        x = self.layer3(x)
        # print('layer3 -----> ',x.shape)
        x = self.layer4(x)
        x = self.pooling2(x)
        # print('layer4 -----> ',x.shape)
        # x = self.final_pooling(x)
        # print('final_pooling -----> ',x.shape)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pooling3(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # print('layer8 -----> ',x.shape)
        x = self.final_pooling(x)
        return x



class TS_UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, frame_num, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(TS_UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = TS_Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = TS_Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = TS_Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        # self.final_pooling = nn.Conv3d(out_channels, out_channels,  (128,1,1), padding=0)
        self.final_pooling = nn.MaxPool3d((frame_num,1,1), stride = (frame_num,1,1))
        self.final_activation = nn.Sigmoid()
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        # final_pooling = nn.MaxPool3d((x.shape[2],1,1), stride = (x.shape[2],1,1))
        # x = final_pooling(x)
        # print(x.shape)
        x = self.final_pooling(x)
        x = self.final_activation(x)

        return x