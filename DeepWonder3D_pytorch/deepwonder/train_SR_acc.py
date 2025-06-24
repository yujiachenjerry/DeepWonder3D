import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import sys
import math
import scipy.io as scio

import torch.nn.functional as F
import numpy as np
from skimage import io
import itertools
import random
import warnings
warnings.filterwarnings("ignore")
import yaml
from torch.cuda.amp import autocast


import deepwonder.SR.SR_data_process_train
from deepwonder.SR.SR_data_process_train import train_preprocess_mean_SR, trainset_mean_SR, train_preprocess_signal_SR, trainset_signal_SR
from deepwonder.SR.SR_net import SR_Net
from deepwonder.SR.SR_network_D import NLayerD

from deepwonder.SR.SR_utils import save_img_train, save_para_dict, UseStyle
#############################################################################################################################################

class train_sr_net_acc():
    def __init__(self, SR_para):
        self.task_type = 'mean' #'signal'
        self.net_type = '' #'signal'
        self.if_D = 0

        self.GPU = '0,1'
        self.SR_n_epochs = 100
        self.SR_batch_size = 4

        self.SR_sub_img_size = 48
        self.SR_img_w = 64
        self.SR_img_h = 64
        self.SR_img_s = 1

        self.SR_lr = 0.00005
        self.SR_b1 = 0.5
        self.SR_b2 = 0.999

        self.SR_f_maps = 16
        self.SR_in_c = 1
        self.SR_out_c = 1

        self.SR_datasets_path = 'datasets'
        self.SR_pth_path = 'pth'
        self.SR_select_img_num = 6000
        self.SR_train_datasets_size = 1000

        self.SR_norm_factor = 1
        self.SR_output_dir = './results'
        self.SR_datasets_folder = 'DataForPytorch'
    
        self.SR_use_pretrain = 0
        self.SR_pretrain_index = '1111'
        self.SR_pretrain_model = '1111'
        self.SR_pretrain_path = 'pth'

        self.SR_sample_up = 15
        self.SR_sample_down = 5
        self.SR_input_pretype = '' 

        self.reset_para(SR_para)
        self.make_folder()
        # self.reset_para(SR_para)

        if self.task_type == 'signal':
            self.SR_in_c = self.SR_img_s
        if self.task_type == 'mean':
            self.SR_in_c = 1
        
        print(UseStyle('SR_f_maps -----> '+str(self.SR_f_maps), mode = 'bold', fore  = 'red'))
        if self.task_type=='mean':
            assert self.SR_f_maps<=16
        if self.task_type=='signal':
            assert self.SR_f_maps>=8
        assert self.SR_sample_up>=self.SR_sample_down




    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time ='SR_HALF_'+self.task_type+'_'+self.net_type+'_up'+str(self.SR_sample_up)\
            +'_down'+str(self.SR_sample_down) +'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")

        self.SR_output_path = self.SR_output_dir + '/' + current_time
        self.SR_pth_save_path = self.SR_pth_path+'//'+ current_time
        print('self.SR_pth_path ---> ',self.SR_pth_path, "self.SR_pth_save_path ---> ",self.SR_pth_save_path)
        if not os.path.exists(self.SR_output_dir): 
            os.mkdir(self.SR_output_dir)
        if not os.path.exists(self.SR_output_path): 
            os.mkdir(self.SR_output_path)
        if not os.path.exists(self.SR_pth_path): 
            os.mkdir(self.SR_pth_path)
        if not os.path.exists(self.SR_pth_save_path): 
            os.mkdir(self.SR_pth_save_path)


    #########################################################################
    #########################################################################
    def reset_para(self, SR_para):
        for key, value in SR_para.items():
            if hasattr(self, key):
                setattr(self, key, value)
        '''
        if self.SR_use_pretrain:
            pretrain_yaml = self.SR_pretrain_path+'//'+self.SR_pretrain_model+'//'+self.task_type+'_SR_para'+'.yaml'
            print('pretrain_yaml -----> ', pretrain_yaml, os.path.exists(pretrain_yaml))
            if os.path.exists(pretrain_yaml):
                # with open(pretrain_yaml, 'r') as f:
                f = open(pretrain_yaml)
                pretrain_para = yaml.load(f.read(), Loader = yaml.FullLoader)
                setattr(self, 'SR_f_maps', pretrain_para['SR_f_maps'])
                setattr(self, 'SR_in_c', pretrain_para['SR_in_c'])
                setattr(self, 'SR_out_c', pretrain_para['SR_out_c'])
                setattr(self, 'SR_norm_factor', pretrain_para['SR_norm_factor'])
                if self.task_type=='signal':
                    setattr(self, 'SR_img_s', pretrain_para['SR_img_s'])
        '''
        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print(self.__dict__)


    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        if 'SR_net' in yaml_dict.keys():
            del yaml_dict['SR_net'] 
        if 'optimizer' in yaml_dict.keys():
            del yaml_dict['optimizer']
        if 'optimizer_D' in yaml_dict.keys():
            del yaml_dict['optimizer_D']
        if 'D_net' in yaml_dict.keys():
            del yaml_dict['D_net'] 
        yaml_name = self.task_type+'_SR_para'+'.yaml'
        save_SR_para_path = self.SR_output_path+ '//'+yaml_name
        save_para_dict(save_SR_para_path, yaml_dict)
        save_SR_para_path = self.SR_pth_save_path+ '//'+yaml_name
        save_para_dict(save_SR_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = self.task_type+'_SR_para_'+str(self.get_netpara(self.SR_net))+'.txt'
        save_SR_para_path = self.SR_output_path+ '//'+txt_name
        save_para_dict(save_SR_para_path, txt_dict)
        save_SR_para_path = self.SR_pth_save_path+ '//'+txt_name
        save_para_dict(save_SR_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def get_netpara(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    #########################################################################
    #########################################################################
    def load_pretrain_para(self, SR_net_model_path):
        model_dict = self.SR_net.state_dict().copy()
        save_model = torch.load(SR_net_model_path).copy()

        # for key in model_dict:
        #     print(key)

        save_model_key1 = list(save_model.keys())[0]
        model_dict_key1 = list(model_dict.keys())[0]
        print(save_model_key1)
        print(model_dict_key1)

        name_para1 = save_model_key1 #'module.net1.conv1.conv1.0.weight'
        if name_para1 in model_dict:
            print(model_dict[name_para1].size())

            print(save_model[name_para1].size())
            rand_para1 = torch.rand(model_dict[name_para1].size()).cuda()

            save_model[name_para1] = rand_para1.clone()
            print(save_model[name_para1].size())
            self.SR_net.load_state_dict(save_model)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list

        # SR_L_Net SR_Net SR_trans_Net

        self.SR_net = SR_Net(net_type = self.net_type,
                            in_ch = self.SR_in_c, 
                            out_ch = self.SR_out_c,
                            f_num = self.SR_f_maps)

        self.SR_net = torch.nn.DataParallel(self.SR_net) 
        # self.SR_net.half()
        self.SR_net.cuda()
        if self.SR_use_pretrain:
            SR_net_pth_name = self.SR_pretrain_index
            SR_net_model_path = self.SR_pretrain_path+'//'+self.SR_pretrain_model+'//'+SR_net_pth_name
            self.load_pretrain_para(SR_net_model_path)

        self.optimizer = torch.optim.Adam(params=itertools.chain(self.SR_net.parameters()), 
                                        lr=self.SR_lr, betas=(self.SR_b1, self.SR_b2),
                                        eps=1e-4)
        if self.if_D:
            self.D_net = NLayerD(self.SR_out_c).cuda()
            self.optimizer_D = torch.optim.Adam(self.D_net.parameters(),
                                                lr=self.SR_lr, betas=(self.SR_b1, self.SR_b2))
        print('Parameters of SR_net -----> ' , self.get_netpara(self.SR_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        if self.task_type == 'mean':
            self.name_list, self.img_list, self.coordinate_list = train_preprocess_mean_SR(self)
        if self.task_type == 'signal':
            self.name_list, self.img_list, self.coordinate_list = train_preprocess_signal_SR(self)


    #########################################################################
    #########################################################################
    def get_SR_inputGT(self, input):
        # print('get_SR_inputGT input -----> ', input.cpu().detach().numpy().max(), input.cpu().detach().numpy().min())
        # input = input/self.SR_norm_factor
        # print('get_SR_inputGT input -----> ', input.cpu().detach().numpy().max(), input.cpu().detach().numpy().min())
        self.up_rate = random.randint(self.SR_sample_down, self.SR_sample_up)
        img_size = self.up_rate*self.SR_sub_img_size
        if self.task_type == 'mean':
            sr_gt = input[:, :, 0:img_size, 0:img_size].clone()
            sr_in1 = input[:, :, 0:img_size, 0:img_size].clone()

            sr_in = F.interpolate(sr_in1, (self.SR_sub_img_size, self.SR_sub_img_size),  mode='bilinear', align_corners=False)
            if len(sr_gt.shape)==3:
                sr_gt = sr_gt.unsqueeze(dim=0)

        if self.task_type == 'signal':
            c_slices = int((self.SR_img_s-1)/2)
            sr_in1 = input[:, :, 0:img_size, 0:img_size].clone()
            sr_in = F.interpolate(sr_in1, (self.SR_sub_img_size, self.SR_sub_img_size),  mode='bilinear', align_corners=False)
            
            # sr_gt = sr_in1[:, c_slices, :, :].clone()
            # sr_gt = sr_in1.clone() # [:, c_slices, :, :].clone()

            if self.SR_out_c==1:
                if_merge_one = 1
            if self.SR_out_c>1:
                if_merge_one = 0

            if if_merge_one:
                sr_gt = sr_in1[:, c_slices, :, :].clone()
            if not if_merge_one:
                sr_gt = sr_in1.clone()
            
            if len(sr_gt.shape)==3:
                sr_gt = sr_gt.unsqueeze(dim=1)
            '''
            delete_num = random.randint(0,3)
            if delete_num==1:
                sr_in[:,0,:,:]=0
                sr_in[:,-1,:,:]=0
            if delete_num==2:
                sr_in[:,0,:,:]=0
                sr_in[:,1,:,:]=0
                sr_in[:,-1,:,:]=0
                sr_in[:,-2,:,:]=0
            '''
            del sr_in1, input
            import gc
            gc.collect()
        return sr_in, sr_gt



    def all_loss(self, sr_gt, sr_out):
        L1_pixelwise = torch.nn.L1Loss().cuda()
        L2_pixelwise = torch.nn.MSELoss().cuda()

        L1_loss_sr = L1_pixelwise(sr_gt, sr_out)
        L2_loss_sr = L2_pixelwise(sr_gt.float(), sr_out.float())
        # print(L1_loss_sr ,L2_loss_sr)
        final_loss = L1_loss_sr +L2_loss_sr

        sr_gt_std = torch.var(sr_gt, dim=1).unsqueeze(1)
        sr_out_std = torch.var(sr_out, dim=1).unsqueeze(1)
        
        if_std_loss = 0
        if if_std_loss:
            # print('sr_gt_std -----> ',sr_gt_std.shape)
            # print('sr_out_std -----> ',sr_out_std.shape)
            L1_loss_std = L1_pixelwise(sr_gt_std, sr_out_std)
            L2_loss_std = L2_pixelwise(sr_gt_std, sr_out_std)
            final_loss = final_loss+L1_loss_std+L2_loss_std
        
        if_mean_loss = 0
        if if_mean_loss:
            sr_gt_mean = torch.mean(sr_gt, dim=1).unsqueeze(1)
            sr_out_mean = torch.mean(sr_out, dim=1).unsqueeze(1)
            # print('sr_gt_mean -----> ',sr_gt_mean.shape)
            # print('sr_out_mean -----> ',sr_out_mean.shape)
            L1_loss_mean = L1_pixelwise(sr_gt-sr_gt_mean, sr_out-sr_out_mean)
            L2_loss_mean = L2_pixelwise(sr_gt-sr_gt_mean, sr_out-sr_out_mean)
            final_loss = final_loss+L1_loss_mean+L2_loss_mean
        
        self.if_D = 0
        if self.if_D:
            out_D = self.D_net(sr_out)
            # print('out_D -----> ',out_D.shape) 
            cuda = True if torch.cuda.is_available() else False
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            gt_label = Variable(Tensor(out_D.shape).fill_(1.0), requires_grad=False)
            out_label = Variable(Tensor(out_D.shape).fill_(0.0), requires_grad=False)
            # print('out_D -----> ',out_D.shape, gt_label.shape)
            loss_G_D = self.BCELoss_function(out_D, gt_label) 
            final_loss = final_loss+loss_G_D*0.01

            gt_D = self.D_net(sr_gt)
            out_D = self.D_net(sr_out.detach()) 
            loss_gt_D = self.BCELoss_function(gt_D, gt_label) 
            loss_out_D = self.BCELoss_function(out_D, out_label)
            loss_D = loss_gt_D + loss_out_D
        if not self.if_D:
            loss_D = final_loss
        #####################################################
        return final_loss ,loss_D #, sr_gt_std, sr_out_std
    

    #########################################################################
    #########################################################################
    def train(self):
        scaler = torch.cuda.amp.GradScaler()
        per_epoch_len = len(self.name_list)
        # L1_pixelwise = torch.nn.L1Loss()
        # L2_pixelwise = torch.nn.MSELoss()
        # L1_pixelwise.cuda()
        # L2_pixelwise.cuda()
        self.BCELoss_function = torch.nn.BCEWithLogitsLoss()
        self.BCELoss_function.cuda()

        prev_time = time.time()
        ########################################################################################################################
        # torch.multiprocessing.set_start_method('spawn')
        sub_img_size = self.SR_sub_img_size
        ########################################################################################################################
        time_start=time.time()
        
        for epoch in range(0, self.SR_n_epochs):
            if self.task_type == 'mean':
                train_data = trainset_mean_SR(self.name_list, self.img_list, self.coordinate_list)
            if self.task_type == 'signal':
                train_data = trainset_signal_SR(self.name_list, self.img_list, self.coordinate_list)
            trainloader = DataLoader(train_data, batch_size=self.SR_batch_size, shuffle=True, pin_memory=False, num_workers=0)

            
            for index, (input, input_name) in enumerate(trainloader):
                
                sr_in, sr_gt = self.get_SR_inputGT(input)
                sr_in = sr_in.half()
                sr_gt = sr_gt.half()
                ####################################################################################################################  
                # time.sleep(1000)
                # sr_in = sr_in.half()
                # sr_gt = sr_gt.half()
                # print('input -----> ', input.cpu().detach().numpy().max(), input.cpu().detach().numpy().min(),
                # 'sr_in -----> ', sr_in.cpu().detach().numpy().max(), sr_in.cpu().detach().numpy().min(),
                # 'sr_gt -----> ', sr_gt.cpu().detach().numpy().max(), sr_gt.cpu().detach().numpy().min())
                with autocast():
                    sr_out, sr_out_da = self.SR_net(sr_in, self.up_rate)
                # print('sr_gt shape -----> ',sr_gt.size())
                # print('sr_in shape -----> ',sr_in.size())
                # print('sr_out shape -----> ',sr_out.size()) , xf2_out, xf3_out
                ####################################################################################################################  
                # L1_loss_sr = L1_pixelwise(sr_gt, sr_out)
                # L2_loss_sr = L2_pixelwise(sr_gt, sr_out)
                # , loss_D, sr_gt_std, sr_out_std #L1_loss_sr + L2_loss_sr

                loss_sr, loss_D = self.all_loss(sr_gt, sr_out) 
                loss_mask_out, loss_mask_out_D = self.all_loss(sr_gt, sr_out_da)  
                # loss_xf1 = self.all_loss(sr_gt, xf1_out)  
                # loss_xf2 = self.all_loss(sr_gt, xf2_out)  
                # loss_xf3 = self.all_loss(sr_gt, xf3_out)  
                ################################################################################################################
                # with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                Total_loss = loss_sr + loss_mask_out # + loss_xf1 # + loss_xf2 + loss_xf3
                # print(Total_loss)
                # Total_loss.half().backward()
                # self.optimizer.step()
                scaler.scale(Total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # self. = 0
                if self.if_D:
                    self.optimizer_D.zero_grad() 
                    # All_loss_D =                            
                    loss_D.backward()                                   
                    self.optimizer_D.step() 
                ################################################################################################################
                batches_done = epoch * per_epoch_len + index + 1
                batches_left = self.SR_n_epochs * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=batches_left/batches_done * (time.time() - prev_time))
                ################################################################################################################
                if index%(100//self.SR_batch_size) == 0:
                    time_end=time.time()
                    print_head = self.task_type.upper()+'_SR_TRAIN'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '    [Epoch %d/%d]   [Batch %d/%d]   [Total_loss: %f]   [Time_Left: %s]'% (
                        epoch,
                        self.SR_n_epochs,
                        index,
                        per_epoch_len,
                        Total_loss, 
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                if (index+1)%(self.SR_train_datasets_size//2//self.SR_batch_size) == 0:
                    # self.SR_train_datasets_size//2//self.SR_batch_size
                    norm_factor = self.SR_norm_factor
                    image_name = input_name

                    '''
                    if self.SR_out_c>1:
                        save_img_train(sr_gt_std, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_gt_std')
                        save_img_train(sr_out_std, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_out_std')

                    print('END input -----> ', input.cpu().detach().numpy().max(), input.cpu().detach().numpy().min(),
                    'sr_in -----> ', sr_in.cpu().detach().numpy().max(), sr_in.cpu().detach().numpy().min(),
                    'sr_gt -----> ', sr_gt.cpu().detach().numpy().max(), sr_gt.cpu().detach().numpy().min())
                    '''
                    save_img_train(sr_in, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_in')
                    save_img_train(sr_gt, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_gt')
                    save_img_train(sr_out, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_out')
                    save_img_train(sr_out_da, self.SR_output_path, epoch, index, input_name, norm_factor, 'sr_out_da')
                    print('sr_out -----> ', sr_out.cpu().detach().numpy().max(), sr_out.cpu().detach().numpy().min())
                    print('sr_in -----> ', sr_in.cpu().detach().numpy().max(), sr_in.cpu().detach().numpy().min())
                    print('sr_gt -----> ', sr_gt.cpu().detach().numpy().max(), sr_gt.cpu().detach().numpy().min())
                    print(Total_loss)

            torch.save(self.SR_net.state_dict(), self.SR_pth_save_path +'//'+ self.task_type+'_SR_' + str(epoch) + '.pth')
            if self.if_D:
                torch.save(self.D_net.state_dict(), self.SR_pth_save_path +'//'+ self.task_type+'_D_' + str(epoch) + '.pth')


    
    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.train()


if __name__ == '__main__':
    SR_parameters={ 'task_type':'signal',  # signal mean
                    'net_type':'ps',  # ps trans_mini2
                    'if_D':0,
                    'GPU':'1',
                    'SR_n_epochs':2000,
                    'SR_batch_size':1,
                    ############################
                    'SR_sub_img_size':32,
                    'SR_img_w':720,
                    'SR_img_h':720,
                    'SR_img_s':5,
                    ############################
                    'SR_lr':0.00001,
                    'SR_b1':0.5,
                    'SR_b2':0.999,
                    ############################
                    'SR_f_maps':16,
                    'SR_in_c':1,
                    'SR_out_c':1,
                    ############################
                    'SR_norm_factor':10,
                    'SR_output_dir':'./results',
                    'SR_datasets_folder':'NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_train/mov_w_bg',  # _only1
                    ############################
                    'SR_datasets_path':'..//datasets',
                    'SR_pth_path':'pth',
                    'SR_select_img_num':10000,
                    'SR_train_datasets_size':500,
                    ############################
                    'SR_use_pretrain':0,
                    'SR_pretrain_index':'signal_SR_480.pth',
                    'SR_pretrain_model':'SR_HALF_mean__up15_down5_202311131524',
                    'SR_pretrain_path':'pth',
                    ############################
                    'SR_sample_up':15,
                    'SR_sample_down':3,
                    'SR_input_pretype':'mean' ,} # 'mean'

    SR_model = train_sr_net_acc(SR_parameters)
    SR_model.run()

