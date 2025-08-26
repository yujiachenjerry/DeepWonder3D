import numpy as np
import argparse
import os
import tifffile as tiff
import time
import datetime
import random
from skimage import io
import logging
import math
import torch
from torch.utils.data import Dataset


def shuffle_datasets_lessMemory(name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    for i in range(0,len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list



def random_transform(noise_patch):
    flip_num = random.randint(0,4)
    if flip_num==1:
        noise_patch = np.flip(noise_patch, 0).copy()
    if flip_num==2:
        noise_patch = np.flip(noise_patch, 1).copy()
    if flip_num==3:
        noise_patch = np.flip(noise_patch, 2).copy()

    rotate_num = random.randint(0,4)
    if rotate_num==1:
        noise_patch = np.rot90(noise_patch, 1, axes=(1, 2)).copy()
    if rotate_num==2:
        noise_patch = np.rot90(noise_patch, 2, axes=(1, 2)).copy()
    if rotate_num==3:
        noise_patch = np.rot90(noise_patch, 3, axes=(1, 2)).copy()


    if abs(np.max(noise_patch))>0:
        # print('abs(np.max(noise_patch)) : ',abs(np.max(noise_patch)))
        rand_bg = np.random.randint(0, abs(np.max(noise_patch))*10000)/10000
    if abs(np.max(noise_patch))==0:
        rand_bg = np.random.randint(0, 10000)/10000

    rand_gama_num = random.randint(0,1)
    if rand_gama_num==0:
        rand_gama = np.random.randint(1000, 2000)/1000
    if rand_gama_num==1:
        rand_gama = np.random.randint(500, 1000)/1000
    # print('rand_gama_num shape -----> ',rand_gama_num)
    noise_patch = (noise_patch+rand_bg)/rand_gama

    return noise_patch



class trainset_mean_SR(Dataset):
    def __init__(self, name_list, img_list, coordinate_list):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.img_list = img_list

    def __getitem__(self, index):
        patch_name = self.name_list[index]
        single_coordinate = self.coordinate_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']

        img_name = single_coordinate['name']
        noise_img = self.img_list[img_name]
        noise_patch1 = noise_img[init_h:end_h, init_w:end_w]
        noise_patch = noise_patch1[np.newaxis,:,:]
        noise_patch = random_transform(noise_patch)

        input = torch.from_numpy(noise_patch).float().cuda()
        return input, patch_name

    def __len__(self):
        return len(self.name_list)



class trainset_signal_SR(Dataset):
    def __init__(self, name_list, img_list, coordinate_list):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.img_list = img_list

    def __getitem__(self, index):
        patch_name = self.name_list[index]
        single_coordinate = self.coordinate_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        img_name = single_coordinate['name']
        noise_img = self.img_list[img_name]
        noise_patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = random_transform(noise_patch)

        input = torch.from_numpy(noise_patch).float().cuda()
        return input, patch_name

    def __len__(self):
        return len(self.name_list)



class testset_mean_SR(Dataset):
    def __init__(self, mean_img, mean_per_coor_list):
        self.mean_img = mean_img
        self.mean_per_coor_list = mean_per_coor_list

    def __getitem__(self, index):
        per_coor = self.mean_per_coor_list[index]
        init_h = per_coor['init_h']
        end_h = per_coor['end_h']
        init_w = per_coor['init_w']
        end_w = per_coor['end_w']

        mean_patch = self.mean_img[init_h:end_h, init_w:end_w]
        # mean_patch = mean_patch1[np.newaxis,:,:]
        mean_im = torch.from_numpy(np.expand_dims(mean_patch.astype(np.float32),0)).cuda().type(torch.FloatTensor)
        return mean_im, per_coor

    def __len__(self):
        return len(self.mean_per_coor_list)
    


class testset_signal_SR(Dataset):
    def __init__(self, img, per_coor_list):
        self.img = img
        self.per_coor_list = per_coor_list

    def __getitem__(self, index):
        per_coor = self.per_coor_list[index]
        init_h = per_coor['init_h']
        end_h = per_coor['end_h']
        init_w = per_coor['init_w']
        end_w = per_coor['end_w']
        init_s = per_coor['init_s']
        end_s = per_coor['end_s']
        noise_patch = self.img[init_s:end_s,init_h:end_h,init_w:end_w]
        # print('noise_patch -----> ',noise_patch.shape)
        im = noise_patch.copy()
        imA = torch.from_numpy(im.astype(np.float32)).cuda().type(torch.FloatTensor)
        # imA = torch.from_numpy(np.expand_dims(imA,0)).cuda()
        return im, per_coor

    def __len__(self):
        return len(self.per_coor_list)
####################################################################################################################################
####################################################################################################################################
def train_preprocess_signal_SR(args):
    img_h = args.SR_img_h
    img_w = args.SR_img_w
    img_s = args.SR_img_s
    print('img_s',img_s)
    norm_factor = args.SR_norm_factor

    datasets_path = args.SR_datasets_path
    datasets_folder = args.SR_datasets_folder

    train_datasets_size = args.SR_train_datasets_size
    select_img_num = args.SR_select_img_num
    input_pretype = args.SR_input_pretype

    im_folder = datasets_path+'//'+datasets_folder
    name_list = []
    coordinate_list={}
    img_list = {}

    print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    print('im_folder -----> ',im_folder)
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    img_num_per_stack = math.ceil(train_datasets_size/stack_num)
    print('stack_num -----> ',stack_num)
    
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir).astype(np.float32)/norm_factor
        
        if im.shape[0]>select_img_num:
            im = im[0:select_img_num,:,:]
        
        # if input_pretype == 'mean':
        #     print('input_pretype == mean')
        #     im_ave_single = np.mean(im, axis=0)
        #     im_ave = np.zeros(im.shape).astype(np.float32)
        #     for i in range(0, im.shape[0]):
        #         im_ave[i,:,:] = im_ave_single
        #     im = im-im_ave
        #     del im_ave
        #     import gc
        #     gc.collect()
        
        im = im - np.min(im)
        im = im/np.max(im)*200
        img_list[im_name] = im
        print('im -----> ', im.max(), im.min())
        whole_w = im.shape[2]
        whole_h = im.shape[1]
        whole_s = im.shape[0]
        for ii in range(0, img_num_per_stack):
            single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0, 'end_s':0}
            init_h = np.random.randint(0,whole_h-img_h)
            end_h = init_h+img_h
            init_w = np.random.randint(0,whole_w-img_w)
            end_w = init_w+img_w
            init_s = np.random.randint(0,whole_s-img_s)
            end_s = init_s+img_s

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w
            single_coordinate['init_s'] = init_s
            single_coordinate['end_s'] = end_s
            single_coordinate['name'] = im_name

            patch_name = datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
            name_list.append(patch_name)
            coordinate_list[patch_name] = single_coordinate
    import sys
    print('name_list ---> ',sys.getsizeof(name_list)/1024/1024)
    print('img_list ---> ',sys.getsizeof(img_list)/1024/1024)
    print('coordinate_list ---> ',sys.getsizeof(coordinate_list)/1024/1024)
    print('im ---> ',sys.getsizeof(im)/1024/1024)
    return  name_list, img_list, coordinate_list



######################################################################################################
def test_preprocess_signal_SR(args):
    batch_size = args.signal_SR_batch_size
    norm_factor = args.signal_SR_norm_factor
    input_pretype = args.signal_SR_input_pretype
    test_datasize = args.signal_SR_test_datasize
    up_rate = args.signal_SR_up_rate

    img_h = args.signal_SR_img_h
    img_w = args.signal_SR_img_w
    img_s2 = args.signal_SR_img_s
    gap_h = args.signal_SR_gap_h
    gap_w = args.signal_SR_gap_w
    gap_s2 = args.signal_SR_gap_s

    img_h_up = args.signal_SR_img_h*up_rate
    img_w_up = args.signal_SR_img_w*up_rate
    gap_h_up = args.signal_SR_gap_h*up_rate
    gap_w_up = args.signal_SR_gap_w*up_rate

    cut_w_up = (img_w - gap_w)/2*up_rate
    cut_h_up = (img_h - gap_h)/2*up_rate
    cut_s = (img_s2 - gap_s2)/2*up_rate

    datasets_folder = args.signal_SR_datasets_folder
    datasets_path = args.signal_SR_datasets_path

    im_folder = datasets_path+'//'+datasets_folder
    # print('im_folder -----> ',im_folder)

    name_list = []
    # train_raw = []
    image_list={}
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        name_list.append(im_name.replace('.tif',''))
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        if im.shape[0]>test_datasize:
            im = im[0:test_datasize,:,:]
        im = (im).astype(np.float32)/norm_factor
        im = im.squeeze()

        # # input_pretype = 'mean'
        # if input_pretype == 'mean':
        #     print('input_pretype == mean')
        #     im_ave_single = np.mean(im, axis=0)
        #     im_ave = np.zeros(im.shape)
        #     for i in range(0, im.shape[0]):
        #         im_ave[i,:,:] = im_ave_single
        #     im = im-im_ave

        image_list[im_name.replace('.tif','')] = im

        print(im.shape)
        whole_w = im.shape[2]
        whole_h = im.shape[1]
        whole_s = im.shape[0]

        whole_w_up = whole_w*up_rate
        whole_h_up = whole_h*up_rate

        if gap_w!=0:
            num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        if gap_w==0:
            num_w = 1
        
        if gap_h!=0:
            num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        if gap_h==0:
            num_h = 1

        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

        # print(whole_s,' ----- ',img_s2,' ----- ',gap_s2)
        # print(whole_h,' ----- ',img_h,' ----- ',gap_h)
        # print(num_w,' ----- ',num_h,' ----- ',num_s)
        
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        per_coor_list = []
        for x in range(0,num_h):
            for y in range(0,num_w):
                for z in range(0,num_s):
                    per_coor = {}
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
                    per_coor['init_h'] = init_h
                    per_coor['end_h'] = end_h
                    per_coor['init_w'] = init_w
                    per_coor['end_w'] = end_w
                    per_coor['init_s'] = init_s
                    per_coor['end_s'] = end_s

                    if y == 0:
                        per_coor['stack_start_w'] = 0
                        per_coor['stack_end_w'] = img_w_up - cut_w_up
                        per_coor['patch_start_w'] = 0
                        per_coor['patch_end_w'] = img_w_up - cut_w_up
                    elif y == num_w-1:
                        per_coor['stack_start_w'] = whole_w_up - img_w_up + cut_w_up
                        per_coor['stack_end_w'] = whole_w_up
                        per_coor['patch_start_w'] = cut_w_up
                        per_coor['patch_end_w'] = img_w_up
                    else:
                        per_coor['stack_start_w'] = y*gap_w_up + cut_w_up
                        per_coor['stack_end_w'] = y*gap_w_up + img_w_up - cut_w_up
                        per_coor['patch_start_w'] = cut_w_up
                        per_coor['patch_end_w'] = img_w_up - cut_w_up

                    if x == 0:
                        per_coor['stack_start_h'] = 0
                        per_coor['stack_end_h'] = img_h_up - cut_h_up
                        per_coor['patch_start_h'] = 0
                        per_coor['patch_end_h'] = img_h_up - cut_h_up
                    elif x == num_h-1:
                        per_coor['stack_start_h'] = whole_h_up - img_h_up + cut_h_up
                        per_coor['stack_end_h'] = whole_h_up
                        per_coor['patch_start_h'] = cut_h_up
                        per_coor['patch_end_h'] = img_h_up
                    else:
                        per_coor['stack_start_h'] = x*gap_h_up + cut_h_up
                        per_coor['stack_end_h'] = x*gap_h_up + img_h_up - cut_h_up
                        per_coor['patch_start_h'] = cut_h_up
                        per_coor['patch_end_h'] = img_h_up - cut_h_up

                    if z == 0:
                        per_coor['stack_start_s'] = z*gap_s2
                        per_coor['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        per_coor['patch_start_s'] = 0
                        per_coor['patch_end_s'] = img_s2-cut_s
                    elif z == num_s-1:
                        per_coor['stack_start_s'] = whole_s-img_s2+cut_s
                        per_coor['stack_end_s'] = whole_s
                        per_coor['patch_start_s'] = cut_s
                        per_coor['patch_end_s'] = img_s2
                    else:
                        per_coor['stack_start_s'] = z*gap_s2+cut_s
                        per_coor['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        per_coor['patch_start_s'] = cut_s
                        per_coor['patch_end_s'] = img_s2-cut_s

                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    per_coor['name'] = im_name.replace('.tif','')
                    # print(' single_coordinate -----> ',single_coordinate)
                    per_coor_list.append(per_coor)
        # print(' per_coor_list -----> ',len(per_coor_list))
        while len(per_coor_list)%batch_size!=0:
            per_coor_list.append(per_coor)
            # print(' per_coor_list -----> ',len(per_coor_list),' -----> ',len(per_coor_list)%args.batch_size,' -----> ',args.batch_size)
        coordinate_list[im_name.replace('.tif','')] = per_coor_list
    return  name_list, image_list, coordinate_list



####################################################################################################################################
def train_preprocess_mean_SR(args):
    img_h = args.SR_img_h
    img_w = args.SR_img_w

    select_img_num = args.SR_select_img_num
    datasets_path = args.SR_datasets_path
    datasets_folder = args.SR_datasets_folder
    train_datasets_size = args.SR_train_datasets_size
    norm_factor = args.SR_norm_factor

    input_pretype = args.SR_input_pretype

    im_folder = datasets_path+'//'+datasets_folder
    name_list = []
    coordinate_list={}
    img_list = {}

    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    img_num_per_stack = math.ceil(train_datasets_size/stack_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir)
        if im.shape[0]>select_img_num:
            im = im[0:select_img_num,:,:]

        im = (im).astype(np.float32)/norm_factor
        if input_pretype == 'mean':
            im_ave_single = np.mean(im, axis=0)

        img_list[im_name] = im_ave_single
        whole_w = im.shape[-1]
        whole_h = im.shape[-2]
        for ii in range(0, img_num_per_stack):
            single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0}
            init_h = np.random.randint(0,whole_h-img_h)
            end_h = init_h+img_h
            init_w = np.random.randint(0,whole_w-img_w)
            end_w = init_w+img_w

            single_coordinate['init_h'] = init_h
            single_coordinate['end_h'] = end_h
            single_coordinate['init_w'] = init_w
            single_coordinate['end_w'] = end_w
            single_coordinate['name'] = im_name

            patch_name = im_name.replace('.tif','')+'_x'+str(init_h)+'_y'+str(init_w)
            name_list.append(patch_name) #datasets_folder+'_'+
            coordinate_list[patch_name] = single_coordinate
    return  name_list, img_list, coordinate_list



######################################################################################################
def test_preprocess_signal_mean_SR(args):
    batch_size = args.batch_size
    signal_SR_norm_factor = args.signal_SR_norm_factor
    mean_SR_norm_factor = args.mean_SR_norm_factor
    signal_SR_input_pretype = args.signal_SR_input_pretype
    mean_SR_input_pretype = args.mean_SR_input_pretype

    test_datasize = args.test_datasize
    up_rate = args.up_rate

    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s

    img_h_up = args.img_h*up_rate
    img_w_up = args.img_w*up_rate
    gap_h_up = args.gap_h*up_rate
    gap_w_up = args.gap_w*up_rate

    cut_w_up = (img_w - gap_w)/2*up_rate
    cut_h_up = (img_h - gap_h)/2*up_rate
    cut_s = (img_s2 - gap_s2)/2*up_rate

    datasets_folder = args.datasets_folder
    datasets_path = args.datasets_path

    im_folder = datasets_path+'//'+datasets_folder
    print('im_folder -----> ',im_folder)

    name_list = []
    image_list={}
    image_mean_list={}
    coordinate_list={}
    mean_coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        name_list.append(im_name.replace('.tif',''))
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        if im.shape[0]>test_datasize:
            im = im[0:test_datasize,:,:]
        im = (im).astype(np.float32)/signal_SR_norm_factor
        im = im.squeeze()

        # # input_pretype = 'mean'
        # if signal_SR_input_pretype == 'mean':
        #     print('input_pretype == mean')
        #     im_ave_single = np.mean(im, axis=0)
        #     im_ave = np.zeros(im.shape)
        #     for i in range(0, im.shape[0]):
        #         im_ave[i,:,:] = im_ave_single
        #     im = im-im_ave

        image_list[im_name.replace('.tif','')] = im
        image_mean_list[im_name.replace('.tif','')] = im_ave_single

        print(im.shape)
        whole_w = im.shape[2]
        whole_h = im.shape[1]
        whole_s = im.shape[0]

        whole_w_up = whole_w*up_rate
        whole_h_up = whole_h*up_rate

        if gap_w!=0:
            num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        if gap_w==0:
            num_w = 1
        if gap_h!=0:
            num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        if gap_h==0:
            num_h = 1
        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

        # print(whole_s,' ----- ',img_s2,' ----- ',gap_s2)
        # print(whole_h,' ----- ',img_h,' ----- ',gap_h)
        # print(num_w,' ----- ',num_h,' ----- ',num_s)
        
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        per_coor_list = []
        for x in range(0,num_h):
            for y in range(0,num_w):
                for z in range(0,num_s):
                    per_coor = {}
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
                    per_coor['init_h'] = init_h
                    per_coor['end_h'] = end_h
                    per_coor['init_w'] = init_w
                    per_coor['end_w'] = end_w
                    per_coor['init_s'] = init_s
                    per_coor['end_s'] = end_s

                    if num_w>1:
                        if y == 0:
                            per_coor['stack_start_w'] = 0
                            per_coor['stack_end_w'] = img_w_up - cut_w_up
                            per_coor['patch_start_w'] = 0
                            per_coor['patch_end_w'] = img_w_up - cut_w_up
                        elif y == num_w-1:
                            per_coor['stack_start_w'] = whole_w_up - img_w_up + cut_w_up
                            per_coor['stack_end_w'] = whole_w_up
                            per_coor['patch_start_w'] = cut_w_up
                            per_coor['patch_end_w'] = img_w_up
                        else:
                            per_coor['stack_start_w'] = y*gap_w_up + cut_w_up
                            per_coor['stack_end_w'] = y*gap_w_up + img_w_up - cut_w_up
                            per_coor['patch_start_w'] = cut_w_up
                            per_coor['patch_end_w'] = img_w_up - cut_w_up
                    if num_w==1:
                        per_coor['stack_start_w'] = 0
                        per_coor['stack_end_w'] = img_w_up
                        per_coor['patch_start_w'] = 0
                        per_coor['patch_end_w'] = img_w_up

                    if num_h>1:
                        if x == 0:
                            per_coor['stack_start_h'] = 0
                            per_coor['stack_end_h'] = img_h_up - cut_h_up
                            per_coor['patch_start_h'] = 0
                            per_coor['patch_end_h'] = img_h_up - cut_h_up
                        elif x == num_h-1:
                            per_coor['stack_start_h'] = whole_h_up - img_h_up + cut_h_up
                            per_coor['stack_end_h'] = whole_h_up
                            per_coor['patch_start_h'] = cut_h_up
                            per_coor['patch_end_h'] = img_h_up
                        else:
                            per_coor['stack_start_h'] = x*gap_h_up + cut_h_up
                            per_coor['stack_end_h'] = x*gap_h_up + img_h_up - cut_h_up
                            per_coor['patch_start_h'] = cut_h_up
                            per_coor['patch_end_h'] = img_h_up - cut_h_up
                    if num_h==1:
                        per_coor['stack_start_h'] = 0
                        per_coor['stack_end_h'] = img_h_up
                        per_coor['patch_start_h'] = 0
                        per_coor['patch_end_h'] = img_h_up
        
                    if z == 0:
                        per_coor['stack_start_s'] = z*gap_s2
                        per_coor['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        per_coor['patch_start_s'] = 0
                        per_coor['patch_end_s'] = img_s2-cut_s
                    elif z == num_s-1:
                        per_coor['stack_start_s'] = whole_s-img_s2+cut_s
                        per_coor['stack_end_s'] = whole_s
                        per_coor['patch_start_s'] = cut_s
                        per_coor['patch_end_s'] = img_s2
                    else:
                        per_coor['stack_start_s'] = z*gap_s2+cut_s
                        per_coor['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        per_coor['patch_start_s'] = cut_s
                        per_coor['patch_end_s'] = img_s2-cut_s

                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    per_coor['name'] = im_name.replace('.tif','')
                    # print(' single_coordinate -----> ',single_coordinate)
                    per_coor_list.append(per_coor)
        # print(' per_coor_list -----> ',len(per_coor_list))
        while len(per_coor_list)%batch_size!=0:
            per_coor_list.append(per_coor)
            # print(' per_coor_list -----> ',len(per_coor_list),' -----> ',len(per_coor_list)%args.batch_size,' -----> ',args.batch_size)
        coordinate_list[im_name.replace('.tif','')] = per_coor_list

        mean_per_coor_list = []
        for x in range(0,num_h):
            for y in range(0,num_w):
                mean_per_coor = {}
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

                mean_per_coor['init_h'] = init_h
                mean_per_coor['end_h'] = end_h
                mean_per_coor['init_w'] = init_w
                mean_per_coor['end_w'] = end_w

                if num_w>1:
                    if y == 0:
                        mean_per_coor['stack_start_w'] = 0
                        mean_per_coor['stack_end_w'] = img_w_up - cut_w_up
                        mean_per_coor['patch_start_w'] = 0
                        mean_per_coor['patch_end_w'] = img_w_up - cut_w_up
                    elif y == num_w-1:
                        mean_per_coor['stack_start_w'] = whole_w_up - img_w_up + cut_w_up
                        mean_per_coor['stack_end_w'] = whole_w_up
                        mean_per_coor['patch_start_w'] = cut_w_up
                        mean_per_coor['patch_end_w'] = img_w_up
                    else:
                        mean_per_coor['stack_start_w'] = y*gap_w_up + cut_w_up
                        mean_per_coor['stack_end_w'] = y*gap_w_up + img_w_up - cut_w_up
                        mean_per_coor['patch_start_w'] = cut_w_up
                        mean_per_coor['patch_end_w'] = img_w_up - cut_w_up
                if num_w==1:
                    mean_per_coor['stack_start_w'] = 0
                    mean_per_coor['stack_end_w'] = img_w_up
                    mean_per_coor['patch_start_w'] = 0
                    mean_per_coor['patch_end_w'] = img_w_up

                if num_h>1:
                    if x == 0:
                        mean_per_coor['stack_start_h'] = 0
                        mean_per_coor['stack_end_h'] = img_h_up - cut_h_up
                        mean_per_coor['patch_start_h'] = 0
                        mean_per_coor['patch_end_h'] = img_h_up - cut_h_up
                    elif x == num_h-1:
                        mean_per_coor['stack_start_h'] = whole_h_up - img_h_up + cut_h_up
                        mean_per_coor['stack_end_h'] = whole_h_up
                        mean_per_coor['patch_start_h'] = cut_h_up
                        mean_per_coor['patch_end_h'] = img_h_up
                    else:
                        mean_per_coor['stack_start_h'] = x*gap_h_up + cut_h_up
                        mean_per_coor['stack_end_h'] = x*gap_h_up + img_h_up - cut_h_up
                        mean_per_coor['patch_start_h'] = cut_h_up
                        mean_per_coor['patch_end_h'] = img_h_up - cut_h_up
                if num_h==1:
                    mean_per_coor['stack_start_h'] = 0
                    mean_per_coor['stack_end_h'] = img_h_up
                    mean_per_coor['patch_start_h'] = 0
                    mean_per_coor['patch_end_h'] = img_h_up

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = datasets_folder+'_x'+str(x)+'_y'+str(y)
                mean_per_coor['name'] = im_name.replace('.tif','')
                # print(' single_coordinate -----> ',single_coordinate)
                mean_per_coor_list.append(mean_per_coor)
        # print(' per_coor_list -----> ',len(per_coor_list))
        while len(mean_per_coor_list)%batch_size!=0:
            mean_per_coor_list.append(mean_per_coor)
            # print(' per_coor_list -----> ',len(per_coor_list),' -----> ',len(per_coor_list)%args.batch_size,' -----> ',args.batch_size)
        mean_coordinate_list[im_name.replace('.tif','')] = mean_per_coor_list
    return  name_list, image_list, image_mean_list, coordinate_list, mean_coordinate_list
