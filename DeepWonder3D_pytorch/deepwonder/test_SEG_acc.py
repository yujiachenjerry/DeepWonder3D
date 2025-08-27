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

from deepwonder.SEG.SEG_data_process_v2 import test_preprocess_lessMemory_SEG, testset_SEG
from deepwonder.SEG.SEG_data_process_v2 import test_preprocess_lessMemory_SEG_lm
from deepwonder.SEG.SEG_network import SEG_Network_3D_Unet
from deepwonder.SEG.SEG_utils import FFDrealign4, inv_FFDrealign4

from deepwonder.utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara, print_dict


#############################################################################################################################################

class test_seg_net_acc():
    def __init__(self, SEG_para):
        self.GPU = '0,1'
        self.SEG_output_dir = './/test_results'
        self.SEG_output_folder = ''

        self.SEG_datasets_folder = ''
        self.SEG_datasets_path = '..//datasets'
        self.SEG_test_datasize = 10000

        self.SEG_img_w = 1
        self.SEG_img_h = 1
        self.SEG_img_s = 1
        self.SEG_batch_size = 4
        self.SEG_select_img_num = 1000

        self.SEG_gap_w = 1
        self.SEG_gap_h = 1
        self.SEG_gap_s = 1

        self.SEG_norm_factor = 1
        self.SEG_pth_path = "pth"
        self.SEG_pth_index = ""
        self.SEG_model = "1111"

        self.SEG_f_maps = 32
        self.SEG_in_c = 1
        self.SEG_out_c = 1

        self.reset_para(SEG_para)
        self.make_folder()

    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = 'SEG_' + self.SEG_datasets_folder + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.SEG_output_path = self.SEG_output_dir + '//' + self.SEG_output_folder  # current_time.replace('//','_')
        if not os.path.exists(self.SEG_output_dir):
            os.mkdir(self.SEG_output_dir)
        if not os.path.exists(self.SEG_output_path):
            os.mkdir(self.SEG_output_path)

    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['SEG_net']
        # del yaml_dict['optimizer'] 
        yaml_name = 'SEG_para' + '.yaml'
        save_SEG_para_path = self.SEG_output_path + '//' + yaml_name
        save_para_dict(save_SEG_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'SEG_para_' + str(get_netpara(self.SEG_net)) + '.txt'
        save_SEG_para_path = self.SEG_output_path + '//' + txt_name
        save_para_dict(save_SEG_para_path, txt_dict)

    #########################################################################
    #########################################################################
    def reset_para(self, SEG_para):
        for key, value in SEG_para.items():
            if hasattr(self, key):
                setattr(self, key, value)

        SEG_yaml = self.SEG_pth_path + '//' + self.SEG_model + '//' + 'SEG_para.yaml'
        if_load_para = 0
        if if_load_para:
            with open(SEG_yaml, "r") as yaml_file:
                siganl_SEG_para = yaml.load(yaml_file, Loader=yaml.FullLoader)
            # print('siganl_SEG_para -----> ',siganl_SEG_para)
            self.SEG_norm_factor = siganl_SEG_para['SEG_norm_factor']
            self.SEG_f_maps = siganl_SEG_para['SEG_f_maps']
            self.SEG_in_c = siganl_SEG_para['SEG_in_c']
            self.SEG_out_c = siganl_SEG_para['SEG_out_c']
            self.SEG_img_s = siganl_SEG_para['SEG_img_s']

        print(UseStyle('Training parameters ----->', mode='bold', fore='red'))
        print_dict(self.__dict__)

    #########################################################################
    #########################################################################
    def initialize_model(self):
        self.if_realign = False
        self.SEG_net = SEG_Network_3D_Unet(UNet_type='Unet4_no_end_UNet3D',
                                           in_channels=self.SEG_in_c,
                                           out_channels=self.SEG_out_c,
                                           f_maps=self.SEG_f_maps,
                                           final_sigmoid=True)
        self.SEG_net = torch.nn.DataParallel(self.SEG_net)

        GPU_list = self.GPU
        print('GPU_list -----> ', GPU_list)
        # print(torch.cuda.current_device())
        if GPU_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list  # str(opt.GPU)
            self.SEG_net.cuda()

        SEG_pth_name = self.SEG_pth_index
        SEG_model_path = self.SEG_pth_path + '//' + self.SEG_model + '//' + SEG_pth_name

        print('SEG_model_path ----->', SEG_model_path, SEG_pth_name)
        # module.
        save_model = torch.load(SEG_model_path)
        for k, v in save_model.items():
            pass
            # print('SAVE MODEDL ----->',k,v.shape)

        now_model = self.SEG_net.state_dict().copy()
        for k, v in self.SEG_net.state_dict().copy().items():
            pass
            # print('NOW MODEDL ----->',k,v.shape)

        # print('save_model KEY 0 -----> ',list(save_model.keys())[0])
        # print('now_model KEY 0 -----> ',list(now_model.keys())[0])
        if list(save_model.keys())[0] == list(now_model.keys())[0]:
            self.SEG_net.load_state_dict(torch.load(SEG_model_path))
            print('LOAD MODEL !!!!!')

        if list(save_model.keys())[0] != list(now_model.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in save_model.items():
                # print('SAVE MODEDL ----->',k)
                # new_name = k[7:]
                new_name = 'module.' + k
                new_state_dict[new_name] = v
            self.SEG_net.load_state_dict(new_state_dict)
            print('LOAD MODEL !!!!!')

        # self.SEG_net.load_state_dict(torch.load(SEG_model_path))
        # print('Parameters of SEG_net -----> ' , self.get_netpara(self.SEG_net) )
        self.save_para()

    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.data_process_lm = 1
        if not self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_list, self.coordinate_list = \
                test_preprocess_lessMemory_SEG(self)
        if self.data_process_lm:
            self.name_list, self.patch_name_list, self.img_dir_list, self.coordinate_list = \
                test_preprocess_lessMemory_SEG_lm(self)
            print('self.img_dir_list -----> ', self.img_dir_list)

    #########################################################################
    #########################################################################
    def test(self):

        prev_time = time.time()
        iteration_num = 0
        # print('c_slices -----> ',c_slices)
        for im_index in range(0, len(self.name_list)):  # opt.test_img_num): #1): #
            im_name = self.name_list[im_index]  # 'minst_4' #
            # print('im_name -----> ', im_name)
            per_coor_list = self.coordinate_list[im_name]
            per_patch_list = self.patch_name_list[im_name]

            if self.data_process_lm:
                img_dir = self.img_dir_list[im_name]
                import tifffile as tiff
                img = tiff.imread(img_dir)
                # img = img-np.min(img)
                max_img_value = np.max(img) * 0  # 0.01
                img[img < max_img_value] = max_img_value
                img = img - np.min(img)
                img = img / np.max(img) * 200

            if not self.data_process_lm:
                img = self.img_list[im_name]

            num_s = math.ceil((img.shape[0] - self.SEG_img_s + self.SEG_gap_s) / self.SEG_gap_s)
            SEG_img = np.zeros((num_s, img.shape[1], img.shape[2]))
            ######################################################################################
            test_SEG_data = testset_SEG(img, per_patch_list, per_coor_list)

            test_SEG_dataloader = DataLoader(test_SEG_data, batch_size=self.SEG_batch_size, shuffle=False,
                                             num_workers=0)
            ######################################################################################
            for iteration, (im, per_coor, patch_name) in enumerate(test_SEG_dataloader):
                # print('im -----> ',im.shape)
                if self.if_realign:
                    im = FFDrealign4(im)
                im = im.cuda()
                # print('im -----> ',im.shape)
                if_padding_input = 1
                if if_padding_input:
                    paddingBottom = 16
                    paddingRight = 16
                    paddingTime = 0
                    im = nn.ReplicationPad2d(
                        (paddingRight, paddingRight, paddingBottom, paddingBottom, paddingTime, paddingTime,))(im)
                SEG_out = self.SEG_net(im.type(torch.FloatTensor))
                # print('SEG_out -----> ',SEG_out.shape)
                if if_padding_input:
                    SEG_out = SEG_out[:, :, paddingTime:SEG_out.shape[2] - paddingTime,
                              paddingRight:SEG_out.shape[3] - paddingRight,
                              paddingBottom:SEG_out.shape[4] - paddingBottom]
                if self.if_realign:
                    SEG_out = inv_FFDrealign4(SEG_out)

                if_print_gpu_use = 0
                if if_print_gpu_use:
                    from deepwonder.utils import get_gpu_mem_info
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
                    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
                          .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

                im_np = im.cpu().detach().numpy()
                SEG_out_np = SEG_out.cpu().detach().numpy()
                # print(np.max(im_np),' ---> ', np.min(im_np),' ---> ', np.max(SEG_out_np),' ---> ', np.min(SEG_out_np))
                # print('SEG_out -----> ',SEG_out.shape)
                ################################################################################################################
                per_epoch_len = len(per_coor_list) // self.SEG_batch_size
                batches_done = im_index * per_epoch_len + iteration + 1
                batches_left = len(self.name_list) * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left / batches_done * (time.time() - prev_time)))
                # prev_time = time.time()
                ################################################################################################################
                if iteration % (1) == 0:
                    time_end = time.time()
                    print_head = 'SEG_TEST'
                    print_head_color = UseStyle(print_head, mode='bold', fore='red')
                    print_body = '   [Batch %d/%d]   [Time_Left: %s]' % (
                        batches_done,
                        batches_left + batches_done,
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore='blue')
                    sys.stdout.write("\r  " + print_head_color + print_body_color)
                ################################################################################################################
                SEG_out = SEG_out.cpu().detach().numpy()
                for SEG_i in range(0, SEG_out.shape[0]):
                    SEG_out_s = np.squeeze(SEG_out[SEG_i, :, :, :])
                    # per_coor_s = per_coor[SEG_i]
                    init_s = int(per_coor['init_s'][SEG_i])
                    stack_start_w = int(per_coor['stack_start_w'][SEG_i])
                    stack_end_w = int(per_coor['stack_end_w'][SEG_i])
                    patch_start_w = int(per_coor['patch_start_w'][SEG_i])
                    patch_end_w = int(per_coor['patch_end_w'][SEG_i])

                    stack_start_h = int(per_coor['stack_start_h'][SEG_i])
                    stack_end_h = int(per_coor['stack_end_h'][SEG_i])
                    patch_start_h = int(per_coor['patch_start_h'][SEG_i])
                    patch_end_h = int(per_coor['patch_end_h'][SEG_i])

                    stack_start_s = int(per_coor['stack_start_s'][SEG_i])
                    # stack_end_s = int(per_coor['stack_end_s'][SEG_i])
                    # patch_start_s = int(per_coor['patch_start_s'][SEG_i])
                    # patch_end_s = int(per_coor['patch_end_s'][SEG_i])
                    # print(stack_start_s)
                    # print('SEG_out_s -----> ',SEG_out_s.shape)
                    # print(stack_start_w, stack_end_w, stack_start_h, stack_end_h)
                    # print(patch_start_w, patch_end_w, patch_start_h, patch_end_h)
                    SEG_img[stack_start_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] = \
                        SEG_out_s[patch_start_h:patch_end_h, patch_start_w:patch_end_w]
            SEG_img = SEG_img * 1000
            save_img(SEG_img, self.SEG_norm_factor, self.SEG_output_path, im_name, tag='', if_nor=0)

    #########################################################################
    #########################################################################
    def run(self):
        print('RUN SEG MODEL =====')
        self.initialize_model()
        self.generate_patch()
        self.test()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    SEG_parameters_test = {'GPU': '0,1',
                           'SEG_output_dir': './/test_results',
                           ###########################
                           'SEG_datasets_folder': 'RMBG_pred_signal_202310231446',
                           'SEG_datasets_path': 'test_results',
                           'SEG_test_datasize': 10000,
                           ###########################
                           'SEG_img_w': 256,
                           'SEG_img_h': 256,
                           'SEG_img_s': 64,
                           'SEG_batch_size': 4,
                           'SEG_select_img_num': 20000,
                           ###########################
                           'SEG_gap_w': 224,
                           'SEG_gap_h': 224,
                           'SEG_gap_s': 32,
                           ###########################
                           'SEG_norm_factor': 1,
                           'SEG_pth_path': "pth//SEG_pth",
                           'SEG_pth_index': "seg_100.pth",
                           'SEG_model': "3DUNetFFD_20231019-1059",
                           ###########################
                           'SEG_f_maps': 4,
                           'SEG_in_c': 4,
                           'SEG_out_c': 4,
                           }

    # current_time = 'SEG_'+self.SEG_datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
    # self.SEG_output_path = self.SEG_output_dir + '//' + 
    SEG_parameters_test['SEG_output_folder'] = 'SEG_' + SEG_parameters_test['SEG_datasets_folder'] + \
                                               '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    SEG_model_test = test_seg_net_acc(SEG_parameters_test)
    SEG_model_test.run()
