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


from deepwonder.SR.SR_data_process import test_preprocess_signal_mean_SR, testset_mean_SR, testset_signal_SR
from deepwonder.SR.SR_net import SR_Net
# from deepwonder.SR.SRnet import SR_trans_Net

from deepwonder.SR.SR_utils import save_img_train, save_para_dict, UseStyle, save_img
from deepwonder.utils import print_dict
#############################################################################################################################################

class test_sr_net_acc():
    def __init__(self, SR_para):
        self.GPU = '0,1'
        self.output_dir = './/test_results'
        self.SR_output_folder = ''

        self.datasets_folder = ''
        self.datasets_path = '..//datasets'
        self.test_datasize = 10000
        self.up_rate = 1

        self.img_w = 1
        self.img_h = 1
        self.img_s = 1
        self.batch_size = 4

        self.signal_SR_img_s = 1
        self.gap_w = 1
        self.gap_h = 1
        self.gap_s = 1

        self.signal_SR_norm_factor = 1
        self.signal_SR_pth_path = "pth"
        self.signal_SR_pth_index = ""
        self.signal_SR_model = "1111"

        self.signal_SR_f_maps = 32
        self.signal_SR_in_c = self.signal_SR_img_s 
        self.signal_SR_out_c = 1
        self.signal_SR_input_pretype = 'mean'
        self.net_type = '' #'signal'

        self.mean_SR_norm_factor = 1
        self.mean_SR_pth_path = "pth"
        self.mean_SR_pth_index = ""
        self.mean_SR_model = "1111"

        self.mean_SR_f_maps = 32
        self.mean_SR_in_c = 1
        self.mean_SR_out_c = 1
        self.mean_SR_input_pretype = 'mean'

        self.reset_para(SR_para)
        self.make_folder()


    #########################################################################
    #########################################################################
    def make_folder(self):
        
        current_time = 'SR_'+self.datasets_folder+'_up'+str(self.up_rate) #+'_'+self.signal_SR_model[-12:]
        self.output_path = self.output_dir + '//' + self.SR_output_folder #current_time
        if not os.path.exists(self.output_dir): 
            os.mkdir(self.output_dir)
        if not os.path.exists(self.output_path): 
            os.mkdir(self.output_path)

        self.result_output_path = self.output_path + '//'
        if not os.path.exists(self.result_output_path): 
            os.mkdir(self.result_output_path)

        self.pred_all_output_path = self.result_output_path # + '/pred_all'
        self.pred_signal_output_path = self.result_output_path  # + '/pred_signal'
        if not os.path.exists(self.pred_all_output_path): 
            os.mkdir(self.pred_all_output_path)
        if not os.path.exists(self.pred_signal_output_path): 
            os.mkdir(self.pred_signal_output_path)


    #########################################################################
    #########################################################################
    def get_netpara(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        del yaml_dict['signal_SR_net'] 
        # del yaml_dict['optimizer'] 
        yaml_name = 'SR_para'+'.yaml'
        save_SR_para_path = self.result_output_path+ '//'+yaml_name
        save_para_dict(save_SR_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'SR_para_'+str(self.get_netpara(self.signal_SR_net))+'.txt'
        save_SR_para_path = self.result_output_path+ '//'+txt_name
        save_para_dict(save_SR_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def reset_para(self, SR_para):
        for key, value in SR_para.items():
            if hasattr(self, key):
                setattr(self, key, value)

        signal_SR_yaml = self.signal_SR_pth_path+'//'+self.signal_SR_model+'//'+'signal_SR_para.yaml'
        
        with open(signal_SR_yaml, "r") as yaml_file:
            siganl_SR_para = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # print('siganl_SR_para -----> ',siganl_SR_para)

        self.signal_SR_norm_factor = siganl_SR_para['SR_norm_factor']
        self.signal_SR_f_maps = siganl_SR_para['SR_f_maps']
        self.signal_SR_in_c = siganl_SR_para['SR_in_c']
        self.signal_SR_out_c = siganl_SR_para['SR_out_c']
        self.signal_SR_input_pretype = siganl_SR_para['SR_input_pretype']
        self.img_s = siganl_SR_para['SR_img_s']
        self.net_type = siganl_SR_para['net_type']

        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print_dict(self.__dict__)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)
        print('GPU_list -----> ',GPU_list)
        # print(torch.cuda.current_device)
        '''
        self.signal_SR_net = SR_Net(in_ch = self.signal_SR_in_c, 
                            out_ch = self.signal_SR_out_c,
                            f_num = self.signal_SR_f_maps)
        '''
        # print('self.signal_SR_in_c ---> ',self.signal_SR_in_c)
        '''
        self.signal_SR_net = SR_trans_Net(in_ch = self.signal_SR_in_c, 
                            out_ch = self.signal_SR_out_c,
                            f_num = self.signal_SR_f_maps)
        '''
        self.signal_SR_net = SR_Net(net_type = self.net_type,
                            in_ch = self.signal_SR_in_c, 
                            out_ch = self.signal_SR_out_c,
                            f_num = self.signal_SR_f_maps)

        self.signal_SR_net = torch.nn.DataParallel(self.signal_SR_net) 
        self.signal_SR_net.cuda()

        signal_SR_pth_name = self.signal_SR_pth_index
        signal_SR_model_path = self.signal_SR_pth_path+'//'+self.signal_SR_model+'//'+signal_SR_pth_name
        self.signal_SR_net.load_state_dict(torch.load(signal_SR_model_path))

        # self.prune_network()
        # print('Parameters of SR_net -----> ' , self.get_netpara(self.signal_SR_net) )
        self.save_para()


    def prune_network(self):
        # for name in self.signal_SR_net.state_dict():
        #     print(name)
        para1 = self.signal_SR_net.state_dict()['module.net.net1.conv1.conv1.0.weight'].cpu().detach().numpy()
        print(para1.shape)
        para1 = para1.reshape(160,3,3)
        save_img(para1, 1, 'test_results', 'net1_conv1_conv1',if_nor=0)

        para2 = self.signal_SR_net.module.net.net1.conv1.conv1[0]
        print(list(para2.named_parameters()))
    # save_img(signal_SR_img, self.signal_SR_norm_factor, self.pred_signal_output_path, im_name,if_nor=1)
    #########################################################################
    #########################################################################
    def generate_patch(self):
        self.name_list, self.img_list, self.mean_img_list, self.coordinate_list, self.mean_coordinate_list = test_preprocess_signal_mean_SR(self)
        print('SR name list : ',self.name_list)

    #########################################################################
    #########################################################################
    def test(self):
        # print('multiprocessing -----> ') , force=True
        # torch.multiprocessing.set_start_method('spawn')
        prev_time = time.time()
        iteration_num = 0
        c_slices = int((self.img_s-1)/2)
        print('c_slices -----> ',c_slices, self.img_s)
        signal_SR_up_rate = self.up_rate
        mean_SR_up_rate = self.up_rate
        for im_index in range(0,len(self.name_list)): #opt.test_img_num): #1): #
            im_name = self.name_list[im_index] #'minst_4' #
            # print('im_name -----> ', im_name)
            per_coor_list = self.coordinate_list[im_name]
            mean_per_coor_list = self.mean_coordinate_list[im_name]
            img = self.img_list[im_name]
            # mean_img = self.mean_img_list[im_name]

            out_h = int(img.shape[1]*signal_SR_up_rate)
            out_w = int(img.shape[2]*signal_SR_up_rate)

            signal_SR_img = np.zeros((img.shape[0], out_h, out_w), dtype='uint16')
            ######################################################################################
            test_signal_sr_data = testset_signal_SR(img, per_coor_list)
            test_signal_sr_dataloader = DataLoader(test_signal_sr_data, batch_size=self.batch_size, shuffle=False,
                                    num_workers=1)
            ######################################################################################
            for iteration, (im, per_coor) in enumerate(test_signal_sr_dataloader): 
                # if signal_SR_up_rate>1:
                SR_out, SR_out_da  = self.signal_SR_net(im.type(torch.FloatTensor) , signal_SR_up_rate)
                # print('im -----> ',im.shape, ' SR_out -----> ',SR_out.shape)
                ################################################################################################################
                per_epoch_len = len(per_coor_list)//self.batch_size
                batches_done = im_index * per_epoch_len + iteration + 1
                batches_left = len(self.name_list) * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left/batches_done * (time.time() - prev_time)))
                if_print_gpu_use = 0
                if if_print_gpu_use:
                    from deepwonder.utils import get_gpu_mem_info
                    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
                    print(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'
                    .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                # prev_time = time.time()
                ################################################################################################################
                if iteration%(1) == 0:
                    time_end=time.time()
                    print_head = 'SR_TEST'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '   [Batch %d/%d]   [Time_Left: %s]'% (
                        batches_done,
                        batches_left+batches_done,
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                SR_out_da = SR_out_da.cpu().detach().numpy()
                for sr_i in range(0, SR_out_da.shape[0]):
                    SR_out_da_s = np.squeeze(SR_out_da[sr_i,:,:,:])
                    # per_coor_s = per_coor[sr_i]
                    init_s = int(per_coor['init_s'][sr_i])
                    stack_start_w = int(per_coor['stack_start_w'][sr_i])
                    stack_end_w = int(per_coor['stack_end_w'][sr_i])
                    patch_start_w = int(per_coor['patch_start_w'][sr_i])
                    patch_end_w = int(per_coor['patch_end_w'][sr_i])

                    stack_start_h = int(per_coor['stack_start_h'][sr_i])
                    stack_end_h = int(per_coor['stack_end_h'][sr_i])
                    patch_start_h = int(per_coor['patch_start_h'][sr_i])
                    patch_end_h = int(per_coor['patch_end_h'][sr_i])
                    # signal_SR_img[init_s+c_slices, stack_start_w:stack_end_w, stack_start_h:stack_end_h] = \
                    # SR_out_da_s[patch_start_w:patch_end_w, patch_start_h:patch_end_h]
                    signal_SR_img[init_s+c_slices, stack_start_h:stack_end_h, stack_start_w:stack_end_w,] = \
                    np.clip(SR_out_da_s[patch_start_h:patch_end_h, patch_start_w:patch_end_w,], 0, 65535).astype('uint16')

            # norm_factor = self.signal_SR_norm_factor
            signal_SR_img = signal_SR_img[self.img_s//2:-self.img_s//2+1,:,:]
            print('signal_SR_img -----> ',signal_SR_img.shape)
            save_img(np.clip(signal_SR_img, 0, 65535).astype('uint16'), self.signal_SR_norm_factor, self.pred_signal_output_path, im_name,if_nor=1)
            # io.imsave(self.pred_signal_output_path + '/' + im_name  + '.tif', np.clip(signal_SR_img, 0, 65535).astype('uint16'))



    #########################################################################
    #########################################################################
    def run(self):
        self.initialize_model()
        self.generate_patch()
        self.test()

        return self.output_path


if __name__ == '__main__':
    SR_parameters_test = { 'GPU' : '1',
                    'output_dir' : './/test_results',
                    'SR_output_folder' : '',
                    ###########################
                    'net_type':'trans',
                    'datasets_path' : '..//datasets',
                    'test_datasize' : 10000,
                    
                    ###########################
                    'img_w' : 24,
                    'img_h' : 24,
                    'img_s' : 5,
                    'batch_size' : 32,
                    ###########################
                    'signal_SR_img_s' : 5,
                    'gap_w' : 12,
                    'gap_h' : 12,
                    'gap_s' : 1,
                    ###########################
                    'signal_SR_norm_factor' : 1,
                    'signal_SR_pth_path' : "pth//SR_pth",

                    'datasets_folder' : 'test_07192023_2_113', 
                     # 1_vid_video_1000_selected vid_video_1000 test_07192023_2
                    'up_rate' : 12,
 
                    # 'signal_SR_pth_index' : "signal_SR_600.pth",  
                    # 'signal_SR_model' : "SR_mean_up15_down5_202309121156", 
                    'signal_SR_pth_index' : "signal_SR_1300.pth",  
                    'signal_SR_model' : "SR_mean_up15_down5_202309201927_k3", 
                    ###########################
                    'signal_SR_f_maps' : 32,
                    'signal_SR_in_c' : 5,
                    'signal_SR_out_c' : 1,
                    'signal_SR_input_pretype' : 'mean',
                    ###########################
                    'mean_SR_norm_factor' : 1,
                    'mean_SR_pth_path' : "pth",
                    'mean_SR_pth_index' : "mean_SR_200.pth",
                    'mean_SR_model' : "Pretrain_SR_mean_up15_down15_202306291101_f8_ks15", 
                    ###########################
                    'mean_SR_f_maps' : 32,
                    'mean_SR_in_c' : 1,
                    'mean_SR_out_c' : 1,
                    'mean_SR_input_pretype' : 'mean',
    }

    SR_parameters_test['SR_output_folder'] = 'SR_'+SR_parameters_test['datasets_folder']\
    +'_up'+str(SR_parameters_test['up_rate'])

    torch.multiprocessing.set_start_method('spawn')
    SR_model_test = test_sr_net_acc(SR_parameters_test)
    SR_model_test.run()
