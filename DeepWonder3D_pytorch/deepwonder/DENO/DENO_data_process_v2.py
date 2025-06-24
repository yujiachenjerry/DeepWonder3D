import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset


def img_remove_time_ave(img):
    noise_im_ave_single = np.mean(img, axis=0)
    noise_im_ave = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        noise_im_ave[i,:,:] = noise_im_ave_single
    img = img-noise_im_ave
    return img

###############################
###############################
class testset_DENO(Dataset):
    def __init__(self, img, per_patch_list, per_coor_list):
        self.per_patch_list = per_patch_list
        self.per_coor_list = per_coor_list
        self.img = img

    def __getitem__(self, index):
        patch_name = self.per_patch_list[index]
        # print('patch_name -----> ',patch_name)
        # print('per_coor_list -----> ',self.per_coor_list)
        per_coor = self.per_coor_list[patch_name]
        init_h = per_coor['init_h']
        end_h = per_coor['end_h']
        init_w = per_coor['init_w']
        end_w = per_coor['end_w']
        init_s = per_coor['init_s']
        end_s = per_coor['end_s']
        img_patch = self.img[init_s:end_s, init_h:end_h, init_w:end_w]
        # img_patch = img_remove_time_ave(img_patch)
        img_patch=torch.from_numpy(np.expand_dims(img_patch, 0))
        #target = self.target[index]
        return img_patch, per_coor, patch_name

    def __len__(self):
        return len(self.per_patch_list)

###############################
###############################
def test_preprocess_lessMemory_DENO(args):
    img_h = args.DENO_img_h
    img_w = args.DENO_img_w
    img_s2 = args.DENO_img_s
    gap_h = args.DENO_gap_h
    gap_w = args.DENO_gap_w
    gap_s2 = args.DENO_gap_s

    input_pretype = args.DENO_input_pretype
    datasets_path = args.DENO_datasets_path
    datasets_folder = args.DENO_datasets_folder
    select_img_num = args.DENO_select_img_num
    normalize_factor = args.DENO_norm_factor

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2

    # print('img_h ',img_h,' img_w ',img_w,' img_s2 ',img_s2,
    #     ' gap_h ',gap_h,' gap_w ',gap_w,' gap_s2 ',gap_s2,
    #     ' cut_w ',cut_w,' cut_h ',cut_h,' cut_s ',cut_s,)

    im_folder = datasets_path+'//'+datasets_folder
    # print('im_folder ----> ',im_folder)
    patch_name_list = {}
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    # print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        name_list.append(im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>select_img_num:
            noise_im = noise_im[-select_img_num:,:,:]

        noise_im = (noise_im).astype(np.float32)/normalize_factor
        # print('normalize_factor ---> ',normalize_factor)
        
        if input_pretype == 'mean':
            noise_im_ave_single = np.mean(noise_im, axis=0)
            noise_im_ave = np.zeros(noise_im.shape)
            for i in range(0, noise_im.shape[0]):
                noise_im_ave[i,:,:] = noise_im_ave_single
            noise_im = noise_im-noise_im_ave
            print('input_pretype == mean')

        '''
        if_norm1 = 0
        if if_norm1:
            noise_im = noise_im-np.min(noise_im)
            norm_rate = 3500
            noise_im = noise_im/np.max(noise_im)*norm_rate-norm_rate//7*2
        else:
            noise_im = noise_im/10
        '''

        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        single_im_coordinate_list, sub_patch_name_list = \
        get_test_patch_list(im_name,
                        whole_w, whole_h, whole_s, 
                        img_w, img_h, img_s2, 
                        gap_w, gap_h, gap_s2, 
                        cut_w, cut_h, cut_s)
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_list, coordinate_list




###############################
###############################
def test_preprocess_lessMemory_DENO_lm(args):
    img_h = args.DENO_img_h
    img_w = args.DENO_img_w
    img_s2 = args.DENO_img_s
    gap_h = args.DENO_gap_h
    gap_w = args.DENO_gap_w
    gap_s2 = args.DENO_gap_s

    input_pretype = args.DENO_input_pretype
    datasets_path = args.DENO_datasets_path
    datasets_folder = args.DENO_datasets_folder
    select_img_num = args.DENO_select_img_num
    normalize_factor = args.DENO_norm_factor

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2

    # print('img_h ',img_h,' img_w ',img_w,' img_s2 ',img_s2,
    #     ' gap_h ',gap_h,' gap_w ',gap_w,' gap_s2 ',gap_s2,
    #     ' cut_w ',cut_w,' cut_h ',cut_h,' cut_s ',cut_s,)

    im_folder = datasets_path+'//'+datasets_folder
    # print('im_folder ----> ',im_folder)
    patch_name_list = {}
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    img_dir_list = {}
    name_list = []

    step_len = 10
    op_index = args.denoise_index
    init_index = (op_index-1)*step_len
    end_index = (op_index)*step_len

    im_name_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    print('im_folder -----> ',im_folder, init_index, end_index)
    print('im_name_list -----> ',im_name_list[init_index: end_index])
    # print('im_name_list -----> ',im_name_list)
    lost_im_name_list = list(os.walk('patch7_txt', topdown=False))[-1][-1]
    print('lost_im_name_list ---> ',lost_im_name_list)

    # for im_name in im_name_list[init_index: end_index]:
    for im_name in lost_im_name_list[init_index: end_index]:
        # print('im_name -----> ',im_name)
        im_name = im_name.replace('.txt','')
        # if '.tif' in im_name:
        print('im_name -----> ',im_name)
        if '_g_0.tif' in im_name or '_g_1.tif' in im_name or '_g_2.tif' in im_name or '_g_3.tif' in im_name or '_g_4.tif' in im_name \
        or '_g_5.tif' in im_name or '_g_6.tif' in im_name or '_g_7.tif' in im_name or '_g_8.tif' in im_name or '_g_9.tif' in im_name:
            
            im_dir = im_folder+'//'+im_name
            '''
            if '.tiff' in im_name:
                im_name = im_name.replace('.tiff','')
            if ('.tif' in im_name)&('.tiff' not in im_name):
                im_name = im_name.replace('.tif','')
            '''
            name_list.append(im_name)
            
            img_dir_list[im_name] = im_dir

            noise_im = tiff.imread(im_dir)
            if noise_im.shape[0]>select_img_num:
                noise_im = noise_im[-select_img_num:,:,:]

            noise_im = (noise_im).astype(np.float32)/normalize_factor
            # print('normalize_factor ---> ',normalize_factor)
            
            whole_w = noise_im.shape[2]
            whole_h = noise_im.shape[1]
            whole_s = noise_im.shape[0]

            del noise_im 

            single_im_coordinate_list, sub_patch_name_list = \
            get_test_patch_list(im_name,
                            whole_w, whole_h, whole_s, 
                            img_w, img_h, img_s2, 
                            gap_w, gap_h, gap_s2, 
                            cut_w, cut_h, cut_s)

            coordinate_list[im_name] = single_im_coordinate_list
            patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_dir_list, coordinate_list



############################################
############################################
def get_test_patch_list(im_name,
                        whole_w, whole_h, whole_s, 
                        img_w, img_h, img_s2, 
                        gap_w, gap_h, gap_s2, 
                        cut_w, cut_h, cut_s):
    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

    # print('num_s ---> ',num_s,'whole_s ---> ',whole_s,'img_s2 ---> ',img_s2,'gap_s2 ---> ',gap_s2)
    single_im_coordinate_list = {}
    sub_patch_name_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2 
                elif z == (num_s-1):
                    init_s = whole_s - img_s2
                    end_s = whole_s 
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = img_w-cut_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x*gap_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = img_h-cut_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z*gap_s2 
                    single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s 
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = img_s2-cut_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s2+cut_s 
                    single_coordinate['stack_end_s'] = whole_s 
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s2
                else:
                    single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s2-cut_s

                patch_name = im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
                sub_patch_name_list.append(patch_name)
                single_im_coordinate_list[patch_name] = single_coordinate
    
    return     single_im_coordinate_list, sub_patch_name_list