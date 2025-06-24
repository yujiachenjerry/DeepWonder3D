import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import functools
import numpy as np

from deepwonder.RMBG.RMBG_utils import FFDrealign4, inv_FFDrealign4
from deepwonder.RMBG.Discriminator import NLayerDiscriminator3D
from deepwonder.RMBG.RMBG_network import Network_3D_Unet
from deepwonder.RMBG.RMBG_data_process_train import train_preprocess_lessMemory_RMBG, trainset_RMBG

from skimage import io

from deepwonder.utils import save_img_train, save_para_dict, UseStyle, get_netpara
import warnings
warnings.filterwarnings("ignore")
import yaml
import sys
########################################################################################################
class train_rmbg_net_acc():
    def __init__(self, RMBG_para):
        self.GPU = '0,1'
        self.RMBG_n_epochs = 100
        self.RMBG_batch_size = 4

        self.RMBG_img_w = 128
        self.RMBG_img_h = 128
        self.RMBG_img_s = 256

        self.RMBG_lr = 0.00005
        self.RMBG_b1 = 0.5
        self.RMBG_b2 = 0.999
        self.RMBG_norm_factor = 1

        self.RMBG_f_maps = 32
        self.RMBG_in_c = 4
        self.RMBG_out_c = 4
        self.RMBG_down_num = 4

        self.RMBG_output_dir = './results'
        self.RMBG_datasets_folder = 'train'
        self.RMBG_GT_folder = 'GT'
        self.RMBG_input_folder = 'Input'  
        self.RMBG_datasets_path = 'datasets'  

        self.RMBG_use_pretrain = 0
        self.RMBG_pretrain_index = 'signal_SR_220.pth'
        self.RMBG_pretrain_model = 'SR_signal_up15_down5_20230302-2250'
        self.RMBG_pretrain_path = 'pth'

        self.RMBG_pth_path = 'pth'
        self.RMBG_select_img_num = 10000  
        self.RMBG_train_datasets_size = 2000  

        self.RMBG_input_pretype = 'mean'
        self.RMBG_GT_pretype = 'min'
        # UNet3D UNet3D_squeeze
        self.RMBG_net_type = 'UNet3D_squeeze'

        self.reset_para(RMBG_para)
        self.make_folder()


    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = 'RMBG_'+self.RMBG_net_type+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.RMBG_output_path = self.RMBG_output_dir + '/' + current_time
        if not os.path.exists(self.RMBG_output_dir): 
            os.mkdir(self.RMBG_output_dir)
        if not os.path.exists(self.RMBG_output_path): 
            os.mkdir(self.RMBG_output_path)

        self.RMBG_pth_save_path = self.RMBG_pth_path+'//'+ current_time
        if not os.path.exists(self.RMBG_pth_path): 
            os.mkdir(self.RMBG_pth_path)
        if not os.path.exists(self.RMBG_pth_save_path): 
            os.mkdir(self.RMBG_pth_save_path)


    #########################################################################
    #########################################################################
    def reset_para(self, RMBG_para):
        for key, value in RMBG_para.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.RMBG_use_pretrain:
            pretrain_yaml = self.RMBG_pretrain_path+'//'+self.RMBG_pretrain_model+'//'+'RMBG_para'+'.yaml'
            print('pretrain_yaml -----> ', pretrain_yaml, os.path.exists(pretrain_yaml))
            if os.path.exists(pretrain_yaml):
                # with open(pretrain_yaml, 'r') as f:
                f = open(pretrain_yaml)
                pretrain_para = yaml.load(f.read(), Loader = yaml.FullLoader)
                setattr(self, 'RMBG_f_maps', pretrain_para['RMBG_f_maps'])
                setattr(self, 'RMBG_in_c', pretrain_para['RMBG_in_c'])
                setattr(self, 'RMBG_out_c', pretrain_para['RMBG_out_c'])
                setattr(self, 'RMBG_norm_factor', pretrain_para['RMBG_norm_factor'])

        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print(self.__dict__)


    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['RMBG_net'] 
        del yaml_dict['optimizer_G'] 
        yaml_name = 'RMBG_para.yaml'
        save_RMBG_para_path = self.RMBG_output_path+ '//'+yaml_name
        save_para_dict(save_RMBG_para_path, yaml_dict)
        save_RMBG_para_path = self.RMBG_pth_save_path+ '//'+yaml_name
        save_para_dict(save_RMBG_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'RMBG_para_'+str(get_netpara(self.RMBG_net))+'.txt'
        save_RMBG_para_path = self.RMBG_output_path+ '//'+txt_name
        save_para_dict(save_RMBG_para_path, txt_dict)
        save_RMBG_para_path = self.RMBG_pth_save_path+ '//'+txt_name
        save_para_dict(save_RMBG_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)

        print('in_channels ---> ',self.RMBG_in_c, ' out_channels ---> ',self.RMBG_out_c)
        self.RMBG_net = Network_3D_Unet(UNet_type = self.RMBG_net_type,
                                            in_channels = self.RMBG_in_c,
                                            out_channels = self.RMBG_out_c,
                                            f_maps=self.RMBG_f_maps,
                                            final_sigmoid = True)
        '''
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
        self.Dnet = NLayerDiscriminator3D(input_nc=self.RMBG_out_c, ndf=16, n_layers=3, norm_layer=norm_layer)
        '''

        self.RMBG_net = torch.nn.DataParallel(self.RMBG_net) 
        self.RMBG_net.cuda()
        # self.Dnet.cuda()
        L1_pixelwise = torch.nn.L1Loss().cuda() 
        L2_pixelwise = torch.nn.MSELoss().cuda()
        Dloss = torch.nn.BCEWithLogitsLoss().cuda()
        if self.RMBG_use_pretrain:
            RMBG_net_pth_name = self.RMBG_pretrain_index
            RMBG_net_model_path = self.RMBG_pretrain_path+'//'+self.RMBG_pretrain_model+'//'+RMBG_net_pth_name
            self.RMBG_net.load_state_dict(torch.load(RMBG_net_model_path))

        self.optimizer_G = torch.optim.Adam(self.RMBG_net.parameters(),
                                        lr=self.RMBG_lr, betas=(self.RMBG_b1, self.RMBG_b2))
        # self.optimizer_D = torch.optim.Adam(self.Dnet.parameters(),
        #                                 lr=self.RMBG_lr, betas=(self.RMBG_b1, self.RMBG_b2))

        print('Parameters of RMBG_net -----> ' , get_netpara(self.RMBG_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.name_list, self.coor_list, self.GT_list, self.raw_list = train_preprocess_lessMemory_RMBG(self)

    #########################################################################
    #########################################################################
    def train(self):
        Tensor = torch.cuda.FloatTensor
        per_epoch_len = len(self.name_list)
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        Dloss = torch.nn.BCEWithLogitsLoss()

        L1_pixelwise.cuda()
        L2_pixelwise.cuda()
        Dloss.cuda()

        prev_time = time.time()
        ########################################################################################################################
        # torch.multiprocessing.set_start_method('spawn')
        ########################################################################################################################
        time_start=time.time()
        for epoch in range(0, self.RMBG_n_epochs):
            train_data = trainset_RMBG(self.name_list, self.coor_list, self.GT_list, self.raw_list)
            trainloader = DataLoader(train_data, batch_size=self.RMBG_batch_size, shuffle=True, num_workers=0, pin_memory=False)
            for index, (input, GT, input_name) in enumerate(trainloader):
                if_use_realign = True
                if if_use_realign:
                    real_A = FFDrealign4(input).cuda()
                    real_B = FFDrealign4(GT).cuda()
                if not if_use_realign:
                    real_A = input
                    real_B = GT
                fake_B = self.RMBG_net(real_A)

                L1_loss = L1_pixelwise(fake_B, real_B)
                L2_loss = L2_pixelwise(fake_B, real_B)

                # print('fake_B -----> ',fake_B.shape)
                '''
                D_fake_B = self.Dnet(fake_B)
                valid = Variable(Tensor(D_fake_B.shape).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(D_fake_B.shape).fill_(0.0), requires_grad=False)
                # print('fake -----> ',fake.shape)
                Dloss1 = Dloss(D_fake_B, fake)
                '''
                ################################################################################################################
                self.optimizer_G.zero_grad() 
                Total_loss = L1_loss + L2_loss #+ Dloss1 
                Total_loss.backward() 
                self.optimizer_G.step() 
                ################################################################################################################
                '''
                D_fake_B = self.Dnet(fake_B.detach())
                D_real_A = self.Dnet(real_A)

                self.optimizer_D.zero_grad()
                real_loss = Dloss(D_real_A, valid)
                fake_loss = Dloss(D_fake_B, fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()
                '''
                ################################################################################################################
                batches_done = epoch * per_epoch_len + index + 1
                batches_left = self.RMBG_n_epochs * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()
                ################################################################################################################
                if index%(100//self.RMBG_batch_size) == 0:
                    time_end=time.time()
                    print_head = 'RMBG_TRAIN'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '    [Epoch %d/%d]   [Batch %d/%d]   [Total_loss: %f]   [Time_Left: %s]'% (
                        epoch,
                        self.RMBG_n_epochs,
                        index,
                        per_epoch_len,
                        Total_loss, 
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                if (index+1)%(2000//self.RMBG_batch_size) == 0:
                    norm_factor = self.RMBG_norm_factor
                    image_name = input_name

                    
                    # if_use_realign = False
                    if if_use_realign:
                        RMBG_out = inv_FFDrealign4(fake_B)
                        RMBG_out = inv_FFDrealign4(fake_B)

                    if not if_use_realign:
                        RMBG_out = fake_B

                    save_img_train(input, self.RMBG_output_path, epoch, index, input_name, norm_factor, 'RMBG_in')
                    save_img_train(GT, self.RMBG_output_path, epoch, index, input_name, norm_factor, 'RMBG_gt')
                    save_img_train(RMBG_out, self.RMBG_output_path, epoch, index, input_name, norm_factor, 'RMBG_out')

            torch.save(self.RMBG_net.state_dict(), self.RMBG_pth_save_path +'//'+'RMBG_' + str(epoch) + '.pth')


    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.train()


if __name__ == '__main__':
    RMBG_parameters={'GPU':'1',
                    'RMBG_n_epochs':300,
                    'RMBG_batch_size':1,
                    ############################
                    'RMBG_img_w':256,
                    'RMBG_img_h':256,
                    'RMBG_img_s':256,
                    ############################
                    'RMBG_lr':0.00005,
                    'RMBG_b1':0.5,
                    'RMBG_b2':0.999,
                    'RMBG_norm_factor':1,
                    'RMBG_f_maps':8,
                    'RMBG_in_c':1,
                    'RMBG_out_c':1,
                    'RMBG_down_num':4,
                    ############################
                    'RMBG_output_dir':'./results',
                    'RMBG_datasets_folder':'NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_train_only1',
                    'RMBG_GT_folder':'mov_wo_bg',
                    'RMBG_input_folder':'mov_w_bg',
                    'RMBG_datasets_path':'..//datasets',
                    ############################
                    'RMBG_use_pretrain':0,
                    'RMBG_pretrain_index':'RMBG_299.pth',
                    'RMBG_pretrain_model':'RMBG_UNet3D_squeeze_202309121346',
                    'RMBG_pretrain_path':'pth',
                    ############################
                    'RMBG_pth_path':'pth',
                    'RMBG_select_img_num':10000,
                    'RMBG_train_datasets_size':2000,
                    ############################
                    'RMBG_input_pretype':'mean',  #'mean' # guassian
                    'RMBG_GT_pretype':'min'}

    RMBG_model = train_rmbg_net_acc(RMBG_parameters)
    RMBG_model.run()
