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

from deepwonder.DENO.DENO_utils import FFDrealign4, inv_FFDrealign4
from deepwonder.DENO.DENO_network import Network_3D_Unet
from deepwonder.DENO.DENO_data_process_v2 import test_preprocess_lessMemory_DENO, testset_DENO
from deepwonder.DENO.DENO_data_process_v2 import test_preprocess_lessMemory_DENO_lm

from deepwonder.utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara, print_dict
#############################################################################################################################################

class test_DENO_net():
    def __init__(self, DENO_para):
        self.GPU = '0,1'
        self.DENO_output_dir = './/test_results'
        self.DENO_output_folder = ''

        self.DENO_datasets_folder = ''
        self.DENO_datasets_path = '..//datasets'
        self.DENO_test_datasize = 10000

        self.DENO_img_w = 1
        self.DENO_img_h = 1
        self.DENO_img_s = 1
        self.DENO_batch_size = 4
        self.DENO_select_img_num = 1000

        self.DENO_gap_w = 1
        self.DENO_gap_h = 1
        self.DENO_gap_s = 1

        self.DENO_norm_factor = 1
        self.DENO_pth_path = "pth"
        self.DENO_pth_index = ""
        self.DENO_model = "1111"

        self.DENO_f_maps = 32
        self.DENO_in_c = 1
        self.DENO_out_c = 1
        self.DENO_input_pretype = 'mean'
        self.denoise_index = 0

        self.reset_para(DENO_para)
        self.make_folder()


    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = 'DENO_'+self.DENO_datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.DENO_output_path = self.DENO_output_dir + '//' + self.DENO_output_folder 
        #current_time.replace('//','_')
        if not os.path.exists(self.DENO_output_dir): 
            os.mkdir(self.DENO_output_dir)
        if not os.path.exists(self.DENO_output_path): 
            os.mkdir(self.DENO_output_path)
    

    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['DENO_net'] 
        # del yaml_dict['optimizer'] 
        yaml_name = 'DENO_para'+'.yaml'
        save_DENO_para_path = self.DENO_output_path+ '//'+yaml_name
        save_para_dict(save_DENO_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'DENO_para_'+str(get_netpara(self.DENO_net))+'.txt'
        save_DENO_para_path = self.DENO_output_path+ '//'+txt_name
        save_para_dict(save_DENO_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def reset_para(self, DENO_para):
        for key, value in DENO_para.items():
            if hasattr(self, key):
                setattr(self, key, value)

        DENO_yaml = self.DENO_pth_path+'//'+self.DENO_model+'//'+'DENO_para.yaml'
        if os.path.exists(DENO_yaml):
            with open(DENO_yaml, "r") as yaml_file:
                siganl_DENO_para = yaml.load(yaml_file, Loader=yaml.FullLoader)
            # print('siganl_DENO_para -----> ',siganl_DENO_para)
            self.DENO_norm_factor = siganl_DENO_para['DENO_norm_factor']
            self.DENO_f_maps = siganl_DENO_para['DENO_f_maps']
            self.DENO_in_c = siganl_DENO_para['DENO_in_c']
            self.DENO_out_c = siganl_DENO_para['DENO_out_c']
            self.DENO_input_pretype = siganl_DENO_para['DENO_input_pretype']

        print(UseStyle('Predict Parameters ----->', mode = 'bold', fore  = 'red'))
        print_dict(self.__dict__)
        # print(self.__dict__)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)

        self.DENO_net = Network_3D_Unet(UNet_type = '3DUNet',
                                            in_channels = self.DENO_in_c,
                                            out_channels = self.DENO_out_c,
                                            f_maps=self.DENO_f_maps,
                                            final_sigmoid = False)
        self.DENO_net = torch.nn.DataParallel(self.DENO_net) 
        self.DENO_net.cuda()

        DENO_pth_name = self.DENO_pth_index
        DENO_model_path = self.DENO_pth_path+'//'+self.DENO_model+'//'+DENO_pth_name

        for para_name, param in self.DENO_net.named_parameters():
            para_name_test = para_name
            # print(para_name)
        model_dict = torch.load(DENO_model_path)

        # for para_name in model_dict:
        #     print(para_name)

        if para_name_test[0:7]=='module':
            self.DENO_net.load_state_dict(torch.load(DENO_model_path))
        else:
            # model_dict = torch.load(DENO_model_path)
            for model_para_name in list(model_dict.keys()): # model_dict:
                # print('model_para_name -----> ',model_para_name)
                new_model_para_name = 'module.'+model_para_name
                model_dict[new_model_para_name] = model_dict.pop(model_para_name)
            self.DENO_net.load_state_dict(model_dict)

        # print('Parameters of DENO_net -----> ' , self.get_netpara(self.DENO_net) )
        self.save_para()

    '''
    list(search_res.keys()):
    for name, param in mymodel.named_parameters():
        print(name)
        print(param.data)
        print("requires_grad:", param.requires_grad)
        print("-----------------------------------")
                para_name_test = para_name
                print(para_name)
    dict = torch.load(model_dir)
    older_val = dict['旧名']
    # 修改参数名
    dict['新名'] = dict.pop('旧名')
    torch.save(dict, './model_changed.pth')
    #验证修改是否成功
    changed_dict = torch.load('./model_changed.pth')
    print(old_val)
    print(changed_dict['新名'])
    '''
    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.data_process_lm = 1
        if not self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_list, self.coordinate_list \
            = test_preprocess_lessMemory_DENO(self)
        if self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_dir_list, self.coordinate_list \
            = test_preprocess_lessMemory_DENO_lm(self)
            # print('self.img_dir_list -----> ',self.img_dir_list)


    #########################################################################
    #########################################################################
    def test(self):
        # torch.multiprocessing.set_start_method('spawn')
        prev_time = time.time()
        iteration_num = 0
        # print('c_slices -----> ',c_slices)
        # print('self.name_list -----> ',self.name_list)
        for im_index in range(0,len(self.name_list)): #opt.test_img_num): #1): #
            im_name = self.name_list[im_index] #'minst_4' #
            # print('im_name -----> ', im_name)
            per_coor_list = self.coordinate_list[im_name]
            per_patch_list = self.patch_name_list[im_name]

            if self.data_process_lm:
                img_dir = self.img_dir_list[im_name]
                import tifffile as tiff
                # print('img_dir ---> ',img_dir)
                img = tiff.imread(img_dir)
                # print('img ---> ',np.max(img), np.min(img))
                img = img-img.mean() #np.mean(img) 
                # print('img ---> ',np.max(img), np.min(img))
                if 1:
                    print('DENO input_pretype == mean')
                    im_ave_single = np.mean(img, axis=0)
                    im_ave = np.zeros(img.shape)
                    for i in range(0, img.shape[0]):
                        im_ave[i,:,:] = im_ave_single
                    img = img-im_ave

            if not self.data_process_lm:
                img = self.img_list[im_name]

            DENO_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            ######################################################################################
            test_DENO_data = testset_DENO(img, per_patch_list, per_coor_list)

            test_DENO_dataloader = DataLoader(test_DENO_data, batch_size=self.DENO_batch_size, shuffle=False,
                                    num_workers=1)
            ######################################################################################
            for iteration, (im, per_coor, patch_name) in enumerate(test_DENO_dataloader): 
                # print(np.max(im.cpu().detach().numpy()), np.min(im.cpu().detach().numpy()))
                im = im.type(torch.FloatTensor)
                # print('im -----> ', im.shape)
                if_padding_input = 1
                if if_padding_input:
                    paddingBottom = 16
                    paddingRight = 16
                    paddingTime = 0
                    im = nn.ReplicationPad2d(( paddingRight, paddingRight, paddingBottom, paddingBottom, paddingTime, paddingTime,))(im)
                DENO_out = self.DENO_net(im)
                # print('im -----> ', im.shape, DENO_out.shape)
                if if_padding_input:
                    DENO_out = DENO_out[:, :, paddingTime:DENO_out.shape[2]-paddingTime, 
                    paddingRight:DENO_out.shape[3]-paddingRight, 
                    paddingBottom:DENO_out.shape[4]-paddingBottom]
                # print('im -----> ', im.shape, DENO_out.shape)
                if_print_gpu_use = 0
                if if_print_gpu_use:
                    from deepwonder.utils import get_gpu_mem_info
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
                    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
                    .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                ################################################################################################################
                per_epoch_len = len(per_coor_list)//self.DENO_batch_size
                batches_done = im_index * per_epoch_len + iteration + 1
                batches_left = len(self.name_list) * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left/batches_done * (time.time() - prev_time)))
                # prev_time = time.time()
                ################################################################################################################
                if iteration%(1) == 0:
                    time_end=time.time()
                    print_head = 'DENO ::: '
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '   [Batch %d/%d]   [Time_Left: %s]'% (
                        batches_done,
                        batches_left+batches_done,
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                DENO_out = DENO_out.cpu().detach().numpy()
                # print(np.max(DENO_out), np.min(DENO_out))
                for DENO_i in range(0, DENO_out.shape[0]):
                    DENO_out_s = np.squeeze(DENO_out[DENO_i,:,:,:])
                    # per_coor_s = per_coor[DENO_i]
                    init_s = int(per_coor['init_s'][DENO_i])
                    stack_start_w = int(per_coor['stack_start_w'][DENO_i])
                    stack_end_w = int(per_coor['stack_end_w'][DENO_i])
                    patch_start_w = int(per_coor['patch_start_w'][DENO_i])
                    patch_end_w = int(per_coor['patch_end_w'][DENO_i])

                    stack_start_h = int(per_coor['stack_start_h'][DENO_i])
                    stack_end_h = int(per_coor['stack_end_h'][DENO_i])
                    patch_start_h = int(per_coor['patch_start_h'][DENO_i])
                    patch_end_h = int(per_coor['patch_end_h'][DENO_i])

                    stack_start_s = int(per_coor['stack_start_s'][DENO_i])
                    stack_end_s = int(per_coor['stack_end_s'][DENO_i])
                    patch_start_s = int(per_coor['patch_start_s'][DENO_i])
                    patch_end_s = int(per_coor['patch_end_s'][DENO_i])
                    DENO_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = \
                    DENO_out_s[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

            # print(np.max(DENO_img), np.min(DENO_img))
            # DENO_img = 
            save_img(DENO_img, self.DENO_norm_factor, self.DENO_output_path, im_name, tag='', if_nor=1)


    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.test()


if __name__ == '__main__':
    DENO_parameters_test = { 'GPU' : '0,1',
                    'DENO_output_dir' : './/test_results',
                    ###########################
                    'DENO_datasets_folder' : 'pred_signal',
                    'DENO_datasets_path' : '..//Deepwonder4//test_results//SR2_test_07192023_2_up12_309201927_k3_113',
                    'DENO_test_datasize' : 10000,
                    ###########################
                    'DENO_img_w' : 320,
                    'DENO_img_h' : 320,
                    'DENO_img_s' : 128,
                    'DENO_batch_size' : 1,
                    'DENO_select_img_num' : 20000,
                    ###########################
                    'DENO_gap_w' : 288,
                    'DENO_gap_h' : 288,
                    'DENO_gap_s' : 96,
                    ###########################
                    'DENO_norm_factor' : 1,
                    'DENO_pth_path' : "pth//DENO_pth",
                    'DENO_pth_index' : "E_20_Iter_5000.pth",   
                    'DENO_model' : "dongyufan3_202312251537",     
                    ###########################
                    'DENO_f_maps' : 16,
                    'DENO_in_c' : 1,
                    'DENO_out_c' : 1,
                    'DENO_input_pretype' : '',
                    'denoise_index' : 0,
    }
    # current_time = 'DENO_'+self.DENO_datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
    # self.DENO_output_path = self.DENO_output_dir + '//' + 
    DENO_parameters_test['DENO_output_folder'] = 'DENO_'+\
    DENO_parameters_test['DENO_datasets_folder']+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")

    torch.multiprocessing.set_start_method('spawn')
    DENO_model_test = test_DENO_net_acc(DENO_parameters_test)
    DENO_model_test.run()