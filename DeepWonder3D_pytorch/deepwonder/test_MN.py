import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import numpy as np
from skimage import io
import math
import tifffile as tiff

from deepwonder.MN.MergeNeuron_SEG import merge_neuron_SEG_mul_inten
from deepwonder.MN.merge_neuron_f import joint_neuron
from deepwonder.MN.merge_neuron_f import listAddcontours_Laplacian_pytorch, list2contours
from deepwonder.MN.merge_neuron_f import listAddtrace, listAdd_remain_trace
import scipy.io as scio

import warnings

warnings.filterwarnings('ignore')
from deepwonder.utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara


#############################################################################
#############################################################################
class calculate_neuron:
    def __init__(self, mn_para):
        self.RMBG_datasets_folder = ''
        self.RMBG_datasets_path = ''

        self.SEG_datasets_folder = ''
        self.SEG_datasets_path = ''

        self.SR_datasets_folder = ''
        self.SR_datasets_path = ''

        self.MN_output_dir = ''
        self.MN_output_folder = ''
        # print('mn_para -----> ',mn_para)
        self.reset_para(mn_para)
        # print('self.SR_datasets_folder -----> ',self.SR_datasets_folder, )
        # print('self.SR_datasets_path -----> ',self.SR_datasets_path, )

        self.make_folder()
        self.save_para()
        self.generate_patch()

    #########################################################################
    #########################################################################
    def reset_para(self, MN_para):
        for key, value in MN_para.items():
            if hasattr(self, key):
                # print('key : ',key)
                setattr(self, key, value)

    #########################################################################
    #########################################################################
    def make_folder(self):
        current_time = 'MN_' + self.RMBG_datasets_folder + '_' + self.SEG_datasets_folder
        self.MN_output_path = self.MN_output_dir + '//' + self.MN_output_folder  # current_time.replace('//','_')
        # print('self.MN_output_path ----> ',self.MN_output_path)
        if not os.path.exists(self.MN_output_dir):
            os.mkdir(self.MN_output_dir)
        if not os.path.exists(self.MN_output_path):
            os.mkdir(self.MN_output_path)

    #########################################################################
    #########################################################################
    def save_para(self):
        yaml_dict = self.__dict__.copy()
        # del yaml_dict['optimizer'] 
        yaml_name = 'MN_para' + '.yaml'
        save_MN_para_path = self.MN_output_path + '//' + yaml_name
        save_para_dict(save_MN_para_path, yaml_dict)

    #########################################################################
    #########################################################################
    def generate_patch(self):
        RMBG_folder = self.RMBG_datasets_path + '//' + self.RMBG_datasets_folder
        # print('\n Merge Neuron Part >>> RMBG_folder : ',RMBG_folder)
        # print(list(os.walk(RMBG_folder, topdown=False))[-1][-1])
        RMBG_name_list = []
        RMBG_list = {}
        for RMBG_name in list(os.walk(RMBG_folder, topdown=False))[-1][-1]:
            # print('im_name -----> ',im_name)
            if '.tif' in RMBG_name:
                RMBG_name_list.append(RMBG_name)
                RMBG_dir = RMBG_folder + '//' + RMBG_name
                # RMBG_im = tiff.imread(RMBG_dir)
                # RMBG_list[RMBG_name] = RMBG_im
                RMBG_list[RMBG_name] = RMBG_dir

        SEG_folder = self.SEG_datasets_path + '//' + self.SEG_datasets_folder
        # print('\n Merge Neuron Part >>> SEG_folder : ',SEG_folder)
        # print('SEG_folder : ',SEG_folder)
        SEG_name_list = []
        SEG_list = {}
        for SEG_name in list(os.walk(SEG_folder, topdown=False))[-1][-1]:
            # print('im_name -----> ',im_name)
            if '.tif' in SEG_name:
                SEG_name_list.append(SEG_name)
                SEG_dir = SEG_folder + '//' + SEG_name
                # SEG_im = tiff.imread(SEG_dir)
                SEG_list[SEG_name] = SEG_dir
                # print(RMBG_name,' ---> ',SEG_dir)

        self.RMBG_name_list = RMBG_name_list
        self.RMBG_list = RMBG_list
        self.SEG_name_list = SEG_name_list
        self.SEG_list = SEG_list

        SR_folder = self.SR_datasets_path + '//' + self.SR_datasets_folder
        # print('SR_folder : ',SR_folder)
        SR_name_list = []
        SR_list = {}
        # SR_folder = '..//results_07082023_exp//SANeuron_07082023_18_view_up5//STEP_3_SR'
        for SR_name in list(os.walk(SR_folder, topdown=False))[-1][-1]:
            # print('im_name -----> ',im_name)
            if '.tif' in SR_name:
                SR_name_list.append(SR_name)
                SR_dir = SR_folder + '//' + SR_name
                # SEG_im = tiff.imread(SEG_dir)
                SR_list[SR_name] = SR_dir
                # print(RMBG_name,' ---> ',SEG_dir)
        self.SR_name_list = SR_name_list
        self.SR_list = SR_list
        # print('RMBG : ',self.RMBG_name_list[0])
        # print('SEG : ',self.SEG_name_list[0])
        # print('SEG : ',self.SEG_list)
        # realign_mov_w_bg_LFM_v_162.0.tif
        # print('SEG : ',SEG_list[self.SEG_name_list[0]])
        return self.RMBG_name_list, self.RMBG_list,
        self.SEG_name_list, self.SEG_list,
        self.SR_name_list, self.SR_list

    #########################################################################
    #########################################################################
    def get_neuron_mask(self, SEG_dir, RMBG_dir, SR_dir, im_name):
        # RMBG_dir = img_RMBG[im_name]
        # SEG_dir = img_SEG[im_name]

        # print('RMBG dir : ',RMBG_dir)
        # print('SEG dir : ',SEG_dir)
        # print('SR dir : ',SR_dir)
        img_SEG = tiff.imread(SEG_dir)
        print('img_SEG ---> ', img_SEG.shape)
        # img_SEG = img_SEG[0:2, :, :]
        img_RMBG = tiff.imread(RMBG_dir)
        img_SR = tiff.imread(SR_dir)

        img_SEG1 = img_SEG.copy()
        img_RMBG1 = img_RMBG.copy()
        # img_SR1 = img_SR.copy()
        #############################################################################
        # whole_mask_list = merge_neuron_SEG_mul(img_SEG1, img_RMBG1)
        self.inten_thres = 0
        whole_mask_list, test_list, all_neuron_mask, all_neuron_remain_mask, \
            mask_stack_filted = merge_neuron_SEG_mul_inten(img_SEG1,
                                                           img_RMBG1,
                                                           quit_round_rate=0.5,
                                                           good_round_rate=0.8,
                                                           good_round_size_rate=0.5,
                                                           corr_mark=0.9,
                                                           max_value=1000,
                                                           if_nmf=False,
                                                           inten_thres=self.inten_thres,
                                                           edge_value=10)
        # print('whole_mask_list -----> ',len(whole_mask_list))
        # print('remain_position ---> ',whole_mask_list[0]['remain_position'][0,:])
        # print('mask -----> ','mask' in whole_mask_list[0])
        # whole_mask_list = listAddcontours_Laplacian(whole_mask_list, img.shape[1], img.shape[2])
        # aaaaaa_name = self.MN_output_path+'//'+im_name+'_f_con.tif'
        # io.imsave(aaaaaa_name, mask_stack_filted)

        # bbbbbb_name = self.MN_output_path+'//'+im_name+'_max.tif'
        # io.imsave(bbbbbb_name, mask_stack_max)

        whole_mask_list = listAddcontours_Laplacian_pytorch(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])

        whole_mask_list = listAddtrace(whole_mask_list, img_RMBG, mode='update', trace_mode='all')
        # print('mask -----> ','mask' in whole_mask_list[0])
        whole_mask_list = listAdd_remain_trace(whole_mask_list, img_RMBG, mode='update', trace_mode='all')
        # print('mask -----> ','mask' in whole_mask_list[0])
        # print(whole_mask_list[0].keys())
        whole_mask_list = listAdd_remain_trace(whole_mask_list, img_SR, dict_name='remain_sr_trace', mode='update',
                                               trace_mode='all')
        # print(whole_mask_list[0].keys())

        # print('mask -----> ','mask' in whole_mask_list[0])
        final_contours, whole_contours = list2contours(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])

        f_con_output_path = self.MN_output_path + '//f_con' + '_' + str(self.inten_thres)
        if not os.path.exists(f_con_output_path):
            os.mkdir(f_con_output_path)
        img_f_contours_name = f_con_output_path + '//' + im_name + '_f_con.tif'
        final_contours = final_contours.clip(0, 65535).astype('uint16')
        io.imsave(img_f_contours_name, final_contours)

        from deepwonder.MN.merge_neuron_f import list2masks
        f_mask_bina_output_path = self.MN_output_path + '//f_mask_bina' + '_' + str(self.inten_thres)
        if not os.path.exists(f_mask_bina_output_path):
            os.mkdir(f_mask_bina_output_path)

        # print('mask -----> ','mask' in whole_mask_list[0])
        final_masks, whole_masks = list2masks(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])
        final_masks_bina = final_masks
        final_masks_bina[final_masks_bina > 0] = 1
        img_f_masks_bina_name = f_mask_bina_output_path + '//' + im_name + '_f_mask_bina.tif'
        final_masks_bina = final_masks_bina.clip(0, 65535).astype('uint16')
        io.imsave(img_f_masks_bina_name, final_masks_bina)

        w_con_output_path = self.MN_output_path + '//w_con' + '_' + str(self.inten_thres)
        if not os.path.exists(w_con_output_path):
            os.mkdir(w_con_output_path)
        img_w_contours_name = w_con_output_path + '//' + im_name + '_w_con.tif'
        whole_contours = whole_contours.clip(0, 65535).astype('uint16')

        # if 0:
        #     if len(whole_contours.shape) == 3:
        #         if len(whole_mask_list) > 0:
        #             io.imsave(img_w_contours_name, whole_contours)
        #
        # if 0:
        #     img_f_contours_name = self.MN_output_path + '//' + im_name + '_mask_stack.tif'
        #     all_neuron_mask = all_neuron_mask.clip(0, 65535).astype('uint16')
        #     io.imsave(img_f_contours_name, all_neuron_mask)
        #
        #     # test_list = listAddtrace(test_list, img_RMBG, mode='update', trace_mode='all')
        #     # test_list = listAddcontours_Laplacian_pytorch(test_list, img_RMBG.shape[1], img_RMBG.shape[2])
        #     # test_contours,test_whole_contours = list2contours(test_list, img_RMBG.shape[1], img_RMBG.shape[2])
        #     test_contours_name = self.MN_output_path + '//' + im_name + '_test_con.tif'
        #     all_neuron_remain_mask = all_neuron_remain_mask.clip(0, 65535).astype('uint16')
        #     io.imsave(test_contours_name, all_neuron_remain_mask)
        #
        #     '''
        #     from deepwonder.MN.merge_neuron_f import list2masks
        #     test_list_masks, whole_masks = list2masks(test_list, img_RMBG.shape[1], img_RMBG.shape[2])
        #     final_masks_bina = final_masks
        #     test_list_masks[test_list_masks>0]=1
        #
        #     img_f_masks_bina_name = self.MN_output_path+'//'+im_name+'_test_f_mask_bina.tif'
        #     final_masks_bina = final_masks_bina.clip(0, 65535).astype('uint16')
        #     io.imsave(img_f_masks_bina_name, test_list_masks)
        #     '''

        mat_output_path = self.MN_output_path + '//mat' + '_' + str(self.inten_thres)
        if not os.path.exists(mat_output_path):
            os.mkdir(mat_output_path)
        mat_save_name = mat_output_path + '//' + im_name + '.mat'
        # data = {'a':whole_mask_list, 'final_contours':final_contours}
        # print('mask -----> ','mask' in whole_mask_list[0])
        # print(whole_mask_list[0]['mask'])
        for i in range(0, len(whole_mask_list)):
            single_neuron = whole_mask_list[i]
            del single_neuron['mask']
        # print('mask -----> ','mask' in whole_mask_list[0])
        # print(whole_mask_list[0].keys())
        scio.savemat(mat_save_name, {'final_mask_list': whole_mask_list, 'final_contours': final_contours})

        img_w_contours_name = self.MN_output_path + '//' + im_name + '.tif'
        whole_contours = whole_contours.clip(0, 65535).astype('uint16')

    #########################################################################
    #########################################################################
    def run(self):
        for im_index in range(0, len(self.RMBG_name_list)):
            im_name = self.RMBG_name_list[im_index]
            # print('im_name : ', im_name)
            img_RMBG = self.RMBG_list[im_name]
            img_SEG = self.SEG_list[im_name]
            img_SR = self.SR_list[im_name]
            self.get_neuron_mask(img_SEG, img_RMBG, img_SR, im_name)


if __name__ == '__main__':
    MN_parameters_test = {'RMBG_datasets_folder': 'RMBG',
                          'RMBG_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          ###########################
                          'SEG_datasets_folder': 'SEG',
                          'SEG_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          ###########################
                          'SR_datasets_folder': 'pred_signal',
                          'SR_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          'MN_output_dir': 'test_results',
                          }

    MN_parameters_test['MN_output_folder'] = 'MN_' + MN_parameters_test['RMBG_datasets_folder'] + \
                                             '_' + MN_parameters_test['SEG_datasets_folder']
    # self.MN_output_path = self.MN_output_dir + '//' + self.MN_output_folder
    MN_model = calculate_neuron(MN_parameters_test)
    MN_model.run()
