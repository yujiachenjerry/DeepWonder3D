import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

from deepwonder.RMBG.RMBG_utils import FFDrealign4, inv_FFDrealign4
from deepwonder.RMBG.RMBG_network import Network_3D_Unet
from deepwonder.RMBG.RMBG_data_process_v2 import test_preprocess_lessMemory_RMBG, testset_RMBG
from deepwonder.RMBG.RMBG_data_process_v2 import test_preprocess_lessMemory_RMBG_lm

from deepwonder.utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara, print_dict
#############################################################################################################################################

class test_rmbg_net_acc():
    def __init__(self, RMBG_para):
        self.GPU = '0,1'
        self.RMBG_output_dir = './/test_results'
        self.RMBG_output_folder = ''

        self.RMBG_datasets_folder = ''
        self.RMBG_datasets_path = '..//datasets'
        self.RMBG_test_datasize = 10000

        self.RMBG_img_w = 1
        self.RMBG_img_h = 1
        self.RMBG_img_s = 1
        self.RMBG_batch_size = 4
        self.RMBG_select_img_num = 1000

        self.RMBG_gap_w = 1
        self.RMBG_gap_h = 1
        self.RMBG_gap_s = 1

        self.RMBG_norm_factor = 1
        self.RMBG_pth_path = "pth"
        self.RMBG_pth_index = ""
        self.RMBG_model = "1111"

        self.RMBG_f_maps = 32
        self.RMBG_in_c = 1
        self.RMBG_out_c = 1
        self.RMBG_input_pretype = 'mean'

        self.reset_para(RMBG_para)
        self.make_folder()


    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = 'RMBG_'+self.RMBG_datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.RMBG_output_path = self.RMBG_output_dir + '//' + self.RMBG_output_folder 
        #current_time.replace('//','_')
        if not os.path.exists(self.RMBG_output_dir): 
            os.mkdir(self.RMBG_output_dir)
        if not os.path.exists(self.RMBG_output_path): 
            os.mkdir(self.RMBG_output_path)
    

    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['RMBG_net'] 
        # del yaml_dict['optimizer'] 
        yaml_name = 'RMBG_para'+'.yaml'
        save_RMBG_para_path = self.RMBG_output_path+ '//'+yaml_name
        save_para_dict(save_RMBG_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'RMBG_para_'+str(get_netpara(self.RMBG_net))+'.txt'
        save_RMBG_para_path = self.RMBG_output_path+ '//'+txt_name
        save_para_dict(save_RMBG_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def reset_para(self, RMBG_para):
        for key, value in RMBG_para.items():
            if hasattr(self, key):
                setattr(self, key, value)

        RMBG_yaml = self.RMBG_pth_path+'//'+self.RMBG_model+'//'+'RMBG_para.yaml'
        with open(RMBG_yaml, "r") as yaml_file:
            siganl_RMBG_para = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # print('siganl_RMBG_para -----> ',siganl_RMBG_para)
        self.RMBG_norm_factor = siganl_RMBG_para['RMBG_norm_factor']
        self.RMBG_f_maps = siganl_RMBG_para['RMBG_f_maps']
        self.RMBG_in_c = siganl_RMBG_para['RMBG_in_c']
        self.RMBG_out_c = siganl_RMBG_para['RMBG_out_c']
        self.RMBG_input_pretype = siganl_RMBG_para['RMBG_input_pretype']

        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print_dict(self.__dict__)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)
        self.if_realign = False
        self.RMBG_net = Network_3D_Unet(UNet_type = 'UNet3D_squeeze',
                                            in_channels = self.RMBG_in_c,
                                            out_channels = self.RMBG_out_c,
                                            f_maps=self.RMBG_f_maps,
                                            final_sigmoid = True)
        self.RMBG_net = torch.nn.DataParallel(self.RMBG_net) 
        self.RMBG_net.cuda()

        RMBG_pth_name = self.RMBG_pth_index
        RMBG_model_path = self.RMBG_pth_path+'//'+self.RMBG_model+'//'+RMBG_pth_name
        self.RMBG_net.load_state_dict(torch.load(RMBG_model_path))
        # print('Parameters of RMBG_net -----> ' , self.get_netpara(self.RMBG_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.data_process_lm = 1
        if not self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_list, self.coordinate_list \
            = test_preprocess_lessMemory_RMBG(self)
        if self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_dir_list, self.coordinate_list \
            = test_preprocess_lessMemory_RMBG_lm(self)
            # print('self.img_dir_list -----> ',self.img_dir_list)


    #########################################################################
    #########################################################################
    def test(self):
        # torch.multiprocessing.set_start_method('spawn')
        prev_time = time.time()
        iteration_num = 0
        # print('c_slices -----> ',c_slices)
        print('self.name_list -----> ',self.name_list)
        for im_index in range(0,len(self.name_list)): #opt.test_img_num): #1): #
            im_name = self.name_list[im_index] #'minst_4' #
            # print('im_name -----> ', im_name)
            per_coor_list = self.coordinate_list[im_name]
            per_patch_list = self.patch_name_list[im_name]

            if self.data_process_lm:
                img_dir = self.img_dir_list[im_name]
                import tifffile as tiff
                img = tiff.imread(img_dir)

                
                # if self.RMBG_input_pretype == 'mean':
                #     noise_im_ave_single = np.mean(img, axis=0)
                #     noise_im_ave = np.zeros(img.shape)
                #     for i in range(0, img.shape[0]):
                #         noise_im_ave[i,:,:] = noise_im_ave_single
                #     img = img-noise_im_ave
                #     print('RMBG input_pretype == mean')
                
                ## RMBG normalize ##
                img = img-np.min(img)
                # norm_rate = 2000
                # img = img/np.max(img)*norm_rate-norm_rate//2

                norm_rate = 300
                img = img/np.max(img)*norm_rate
                # from deepwonder.RMBG.RMBG_data_process_v2 import img_remove_time_ave
                # img = img_remove_time_ave(img)


            if not self.data_process_lm:
                img = self.img_list[im_name]

            RMBG_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='uint16')
            ######################################################################################
            test_RMBG_data = testset_RMBG(img, per_patch_list, per_coor_list)

            test_RMBG_dataloader = DataLoader(test_RMBG_data, batch_size=self.RMBG_batch_size, shuffle=False,
                                    num_workers=1)
            ######################################################################################
            for iteration, (im, per_coor, patch_name) in enumerate(test_RMBG_dataloader): 
                # print('im -----> ',im.shape)
                if self.if_realign:
                    im = FFDrealign4(im)
                if_padding_input = 1
                if if_padding_input:
                    paddingBottom = 16
                    paddingRight = 16
                    paddingTime = 0
                    im = nn.ReplicationPad2d(( paddingRight, paddingRight, paddingBottom, paddingBottom, paddingTime, paddingTime,))(im)

                RMBG_out = self.RMBG_net(im.type(torch.FloatTensor))

                if if_padding_input:
                    RMBG_out = RMBG_out[:, :, paddingTime:RMBG_out.shape[2]-paddingTime, 
                    paddingRight:RMBG_out.shape[3]-paddingRight, 
                    paddingBottom:RMBG_out.shape[4]-paddingBottom]
                if self.if_realign:
                    RMBG_out = inv_FFDrealign4(RMBG_out)

                if_print_gpu_use = 0
                if if_print_gpu_use:
                    from deepwonder.utils import get_gpu_mem_info
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
                    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
                    .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                ################################################################################################################
                per_epoch_len = len(per_coor_list)//self.RMBG_batch_size
                batches_done = im_index * per_epoch_len + iteration + 1
                batches_left = len(self.name_list) * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left/batches_done * (time.time() - prev_time)))
                # prev_time = time.time()
                ################################################################################################################
                if iteration%(1) == 0:
                    time_end=time.time()
                    print_head = 'RMBG_TEST'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '   [Batch %d/%d]   [Time_Left: %s]'% (
                        batches_done,
                        batches_left+batches_done,
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                RMBG_out = RMBG_out.cpu().detach().numpy()
                for RMBG_i in range(0, RMBG_out.shape[0]):
                    RMBG_out_s = np.squeeze(RMBG_out[RMBG_i,:,:,:])
                    # per_coor_s = per_coor[RMBG_i]
                    init_s = int(per_coor['init_s'][RMBG_i])
                    stack_start_w = int(per_coor['stack_start_w'][RMBG_i])
                    stack_end_w = int(per_coor['stack_end_w'][RMBG_i])
                    patch_start_w = int(per_coor['patch_start_w'][RMBG_i])
                    patch_end_w = int(per_coor['patch_end_w'][RMBG_i])

                    stack_start_h = int(per_coor['stack_start_h'][RMBG_i])
                    stack_end_h = int(per_coor['stack_end_h'][RMBG_i])
                    patch_start_h = int(per_coor['patch_start_h'][RMBG_i])
                    patch_end_h = int(per_coor['patch_end_h'][RMBG_i])

                    stack_start_s = int(per_coor['stack_start_s'][RMBG_i])
                    stack_end_s = int(per_coor['stack_end_s'][RMBG_i])
                    patch_start_s = int(per_coor['patch_start_s'][RMBG_i])
                    patch_end_s = int(per_coor['patch_end_s'][RMBG_i])
                    RMBG_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = \
                    np.clip(RMBG_out_s[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w], 0, 65535).astype('uint16')

            # print(np.max(RMBG_img), np.min(RMBG_img))
            # RMBG_img = 
            save_img(np.clip(RMBG_img, 0, 65535).astype('uint16'), 
                     self.RMBG_norm_factor, 
                     self.RMBG_output_path, 
                     im_name, tag='', if_nor=0,
                     if_filter_ourliers=1, 
                     ourliers_thres = 65000)


    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.test()


if __name__ == '__main__':
    RMBG_parameters_test = { 'GPU' : '0,1',
                    'RMBG_output_dir' : './/test_results',
                    ###########################
                    'RMBG_datasets_folder' : 'pred_signal',
                    'RMBG_datasets_path' : '..//Deepwonder4//test_results//SR2_test_07192023_2_up12_309201927_k3_113',
                    'RMBG_test_datasize' : 10000,
                    ###########################
                    'RMBG_img_w' : 320,
                    'RMBG_img_h' : 320,
                    'RMBG_img_s' : 128,
                    'RMBG_batch_size' : 1,
                    'RMBG_select_img_num' : 20000,
                    ###########################
                    'RMBG_gap_w' : 288,
                    'RMBG_gap_h' : 288,
                    'RMBG_gap_s' : 96,
                    ###########################
                    'RMBG_norm_factor' : 1,
                    'RMBG_pth_path' : "pth//RMBG_pth",
                    'RMBG_pth_index' : "RMBG_77.pth",   
                    'RMBG_model' : "RMBG_UNet3D_squeeze_202309121346",     
                    ###########################
                    'RMBG_f_maps' : 32,
                    'RMBG_in_c' : 4,
                    'RMBG_out_c' : 4,
                    'RMBG_input_pretype' : '',
    }
    # current_time = 'RMBG_'+self.RMBG_datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
    # self.RMBG_output_path = self.RMBG_output_dir + '//' + 
    RMBG_parameters_test['RMBG_output_folder'] = 'RMBG_'+\
    RMBG_parameters_test['RMBG_datasets_folder']+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")

    torch.multiprocessing.set_start_method('spawn')
    RMBG_model_test = test_rmbg_net_acc(RMBG_parameters_test)
    RMBG_model_test.run()