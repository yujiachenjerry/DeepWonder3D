import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset

###
#  this file is for pre-processing images for training and test.
#  currently we use splitted dataset


###############################
# dataset class
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

    if_rand_inten = 1
    if if_rand_inten:
        rand_bg = np.random.randint(0, np.max(noise_patch))
        rand_gama_num = random.randint(0,1)
        if rand_gama_num==0:
            rand_gama = np.random.randint(1000, 2000)/1000
        if rand_gama_num==1:
            rand_gama = np.random.randint(500, 1000)/1000
        # print('rand_gama_num shape -----> ',rand_gama_num)
        noise_patch = (noise_patch+rand_bg)/rand_gama
    return noise_patch


from torch.nn import functional as F
def random_imresize(noise_patch, sample_down=1, SR_sample_up=10):
    img_size = noise_patch.shape[-1]
    up_rate = random.randint(sample_down, SR_sample_up)
    sub_img_size = img_size//up_rate

    # sub_noise_patch = np.zeros((noise_patch.shape[0], sub_img_size, sub_img_size))
    # resize_noise_patch = np.zeros((noise_patch.shape[0], noise_patch.shape[1], noise_patch.shape[2]))
    '''
    sub_noise_patch = torch.zeros((noise_patch.shape[0], noise_patch.shape[1], sub_img_size, sub_img_size))
    resize_noise_patch = torch.zeros((noise_patch.shape[0], noise_patch.shape[1], noise_patch.shape[2], noise_patch.shape[3]))
    for i in range(0, noise_patch.shape[0]):
        for ii in range(0, noise_patch.shape[1]):
            noise_patch_s = noise_patch[i,ii,:,:]
            print('noise_patch_s -----> ',noise_patch_s.shape)
            sub_noise_patch[i,ii,:,:] = F.interpolate(noise_patch_s, (sub_img_size, sub_img_size),  mode='bilinear', align_corners=False)
            resize_noise_patch[i,ii,:,:] = F.interpolate(sub_noise_patch[i,:,:], (img_size, img_size),  mode='bilinear', align_corners=False)
    '''
    sub_noise_patch = F.interpolate(noise_patch, (sub_img_size, sub_img_size),  mode='bilinear', align_corners=False)
    # print('sub_noise_patch -----> ',sub_noise_patch.shape)
    resize_noise_patch = F.interpolate(sub_noise_patch, (img_size, img_size),  mode='bilinear', align_corners=False)
    # print('resize_noise_patch -----> ',resize_noise_patch.shape)
    return resize_noise_patch


class trainset_RMBG(Dataset):
    def __init__(self, name_list, coor_list, GT_list, raw_list):
        self.name_list = name_list
        self.coor_list=coor_list
        self.GT_list = GT_list
        self.raw_list = raw_list

    def __getitem__(self, index):
        #fn = self.images[index]
        patch_name = self.name_list[index]
        single_coordinate = self.coor_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        name = single_coordinate['name']
        raw = self.raw_list[name]
        GT = self.GT_list[name]
        input = raw[init_s: end_s, init_h:end_h, init_w:end_w]
        target = GT[init_s: end_s, init_h:end_h, init_w:end_w]

        if_random_imresize = 1
        if_random_transform = 1
        if if_random_transform:
            input = random_transform(input)
            # print('input -----> ',input.shape)

        input=torch.from_numpy(np.expand_dims(input, 0)).cuda().float()
        target=torch.from_numpy(np.expand_dims(target, 0)).cuda().float()

        if if_random_imresize:
            input = random_imresize(input)
            # print('input -----> ',input.shape)
        #target = self.target[index]
        return input, target, patch_name

    def __len__(self):
        return len(self.name_list)


class testset_RMBG(Dataset):
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
        img_patch=torch.from_numpy(np.expand_dims(img_patch, 0))
        #target = self.target[index]
        return img_patch, per_coor, patch_name

    def __len__(self):
        return len(self.per_patch_list)



###############################
###############################
class testset_RMBG2D(Dataset):
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
        # img_patch=torch.from_numpy(np.expand_dims(img_patch, 0))
        img_patch=torch.from_numpy(img_patch)
        #target = self.target[index]
        return img_patch, per_coor, patch_name

    def __len__(self):
        return len(self.per_patch_list)



###############################
# pre-process
def get_gap_s(args, img, stack_num):
    whole_w = img.shape[2]
    whole_h = img.shape[1]
    whole_s = img.shape[0]
    # print('whole_w -----> ',whole_w)
    # print('whole_h -----> ',whole_h)
    # print('whole_s -----> ',whole_s)
    w_num = math.floor((whole_w-args.img_w)/args.gap_w)+1
    h_num = math.floor((whole_h-args.img_h)/args.gap_h)+1
    s_num = math.ceil(args.train_datasets_size/w_num/h_num/stack_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    gap_s = math.floor((whole_s-args.img_s*2)/(s_num-1))
    # print('gap_s -----> ',gap_s)
    return gap_s


###############################
###############################
def test_preprocess_lessMemory_RMBG(args):
    img_h = args.RMBG_img_h
    img_w = args.RMBG_img_w
    img_s2 = args.RMBG_img_s
    gap_h = args.RMBG_gap_h
    gap_w = args.RMBG_gap_w
    gap_s2 = args.RMBG_gap_s

    input_pretype = args.RMBG_input_pretype
    datasets_path = args.RMBG_datasets_path
    datasets_folder = args.RMBG_datasets_folder
    select_img_num = args.RMBG_select_img_num
    normalize_factor = args.RMBG_norm_factor

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2

    print('img_h ',img_h,' img_w ',img_w,' img_s2 ',img_s2,
        ' gap_h ',gap_h,' gap_w ',gap_w,' gap_s2 ',gap_s2,
        ' cut_w ',cut_w,' cut_h ',cut_h,' cut_s ',cut_s,)

    # datasets//NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_test//mov_wo_bg
    # datasets//NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_test//mov_w_bg
    im_folder = datasets_path+'//'+datasets_folder
    print('im_folder ----> ',im_folder)
    patch_name_list = {}
    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = {}

    print('list(os.walk(im_folder, topdown=False)) -----> ',list(os.walk(im_folder, topdown=False)))
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        name_list.append(im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>select_img_num:
            noise_im = noise_im[-select_img_num:,:,:]

        noise_im = (noise_im).astype(np.float32)/normalize_factor
        
        # if input_pretype == 'mean':
        #     noise_im_ave_single = np.mean(noise_im, axis=0)
        #     noise_im_ave = np.zeros(noise_im.shape)
        #     for i in range(0, noise_im.shape[0]):
        #         noise_im_ave[i,:,:] = noise_im_ave_single
        #     noise_im = noise_im-noise_im_ave

        img_list[im_name] = noise_im

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)

        print('num_s ---> ',num_s,'whole_s ---> ',whole_s,'img_s2 ---> ',img_s2,'gap_s2 ---> ',gap_s2)
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
        coordinate_list[im_name] = single_im_coordinate_list
        patch_name_list[im_name] = sub_patch_name_list
    return  name_list, patch_name_list, img_list, coordinate_list


def generate_gaussian_kernel(kernel_size):
    sigma = 1.0  # 高斯核的标准差
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    # 生成二维高斯卷积核
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    # 对卷积核进行归一化
    kernel /= np.sum(kernel)
    return kernel



def train_preprocess_lessMemory_RMBG(args):
    input_pretype = args.RMBG_input_pretype
    GT_pretype = args.RMBG_GT_pretype

    datasets_path = args.RMBG_datasets_path
    datasets_folder = args.RMBG_datasets_folder
    GT_folder = args.RMBG_GT_folder
    input_folder = args.RMBG_input_folder

    img_w = args.RMBG_img_w
    img_h = args.RMBG_img_h
    img_s = args.RMBG_img_s

    normalize_factor = args.RMBG_norm_factor
    train_datasets_size = args.RMBG_train_datasets_size
    ###############################
    name_list = []
    patch_name_list = []
    GT_list={}
    input_list={}
    GT_folder = datasets_path+'//'+datasets_folder+'//'+GT_folder
    input_folder = datasets_path+'//'+datasets_folder+'//'+input_folder
    print('input_folder -----> ',input_folder)

    img_num = len(list(os.walk(input_folder, topdown=False))[-1][-1])
    for i in range(0, img_num):
        input_name = list(os.walk(input_folder, topdown=False))[-1][-1][i]
        GT_name = list(os.walk(GT_folder, topdown=False))[-1][-1][i]
        
        input_dir = input_folder+'//'+input_name
        GT_dir = GT_folder+'//'+GT_name
        print('read im_name -----> ',input_dir)
        print('read im_name -----> ',GT_dir)
        input_img = tiff.imread(input_dir)
        GT_img = tiff.imread(GT_dir)
        # im = im.transpose(2,0,1)
        print(input_img.shape)

        # GT_img = np.expand_dims(GT_img, axis=0)
        # input_img = input_img.transpose(2,0,1)
        input_img = input_img.astype(np.float32)/normalize_factor

        # if input_pretype == 'mean':
        #     print('input_pretype == mean')
        #     input_img_ave_single = np.mean(input_img, axis=0)
        #     input_img_ave = np.zeros(input_img.shape)
        #     for i in range(0, input_img.shape[0]):
        #         input_img_ave[i,:,:] = input_img_ave_single
        #     input_img = input_img-input_img_ave

        # input_pretype = 'guassian'
        import torch.nn as nn
        if input_pretype == 'guassian':
            print('input_pretype == guassian')
            g_k_size = 15
            g_k = generate_gaussian_kernel(g_k_size)

            g_k_tensor = torch.FloatTensor(g_k).unsqueeze(0).unsqueeze(0)
            g_k_tensor = nn.Parameter(data=g_k_tensor, requires_grad=False).cuda()

            input_img_tensor = torch.FloatTensor(input_img).unsqueeze(0)
            input_img_tensor = nn.Parameter(data=input_img_tensor, requires_grad=False).cuda()
            print('input_img_tensor ---> ',input_img_tensor.shape)
            # input_img_ave_single = np.mean(input_img, axis=0)
            p_size = g_k_size//2
            input_img_tensor_p = nn.ReplicationPad2d((p_size, p_size, p_size, p_size))(input_img_tensor)
            print('input_img_tensor_p ---> ',input_img_tensor_p.shape)
            input_img_g = np.zeros(input_img.shape)
            for i in range(0, input_img.shape[0]):
                aaa = F.conv2d(input_img_tensor_p[0,i,:,:].unsqueeze(0), g_k_tensor, padding=0)
                # print('aaa ---> ',aaa.shape)
                input_img_g[i,:,:] = aaa.cpu().detach().numpy()
            print('input_img_g ---> ',input_img_g.shape)
            input_img = input_img-input_img_g
        # input_img = input_img-np.min(input_img)
        # norm_rate = 2000
        # input_img = input_img/np.max(input_img)*norm_rate-norm_rate//2

        input_img = input_img-np.min(input_img)
        norm_rate = 300
        input_img = input_img/np.max(input_img)*norm_rate
        # img_list[im_name] = noise_im

        if GT_pretype == 'min':
            print('GT_pretype == min')
            GT_img_ave_single = np.min(GT_img, axis=0)
            GT_img_ave = np.zeros(GT_img.shape,dtype='float32')
            for i in range(0, GT_img.shape[0]):
                GT_img_ave[i,:,:] = GT_img_ave_single
            GT_img = GT_img-GT_img_ave

        GT_list[GT_name.replace('.tif','')] = GT_img
        input_list[input_name.replace('.tif','')] = input_img
        name_list.append(GT_name.replace('.tif',''))
    
    num_per_img = math.ceil(train_datasets_size/img_num)
    coor_list = {}
    for i in range(0, img_num):
        for ii in range(0, num_per_img):
            per_coor = {}
            init_w = np.random.randint(0, GT_img.shape[-2]-img_w-1)
            init_h = np.random.randint(0, GT_img.shape[-1]-img_h-1)
            init_s = np.random.randint(0, GT_img.shape[0]-img_s-1)

            per_coor['name'] = name_list[i]
            per_coor['init_w'] = init_w
            per_coor['init_h'] = init_h
            per_coor['init_s'] = init_s

            per_coor['end_w'] = init_w+img_w
            per_coor['end_h'] = init_h+img_h
            per_coor['end_s'] = init_s+img_s
            per_coor['name'] = name_list[i]

            patch_name = name_list[i]+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
            patch_name_list.append(patch_name.replace('.tif',''))
            coor_list[patch_name] = per_coor
    return  patch_name_list, coor_list, GT_list, input_list


##############################################################################
##############################################################################
##############################################################################
def train_preprocess_lessMemory_RMBG2D(args):
    input_pretype = args.RMBG_input_pretype
    GT_pretype = args.RMBG_GT_pretype

    datasets_path = args.RMBG_datasets_path
    datasets_folder = args.RMBG_datasets_folder
    GT_folder = args.RMBG_GT_folder
    input_folder = args.RMBG_input_folder

    img_w = args.RMBG_img_w
    img_h = args.RMBG_img_h
    img_s = args.RMBG_img_s

    normalize_factor = args.RMBG_norm_factor
    train_datasets_size = args.RMBG_train_datasets_size
    ###############################
    name_list = []
    patch_name_list = []
    GT_list={}
    input_list={}
    GT_folder = datasets_path+'//'+datasets_folder+'//'+GT_folder
    input_folder = datasets_path+'//'+datasets_folder+'//'+input_folder
    print('input_folder -----> ',input_folder)

    img_num = len(list(os.walk(input_folder, topdown=False))[-1][-1])
    for i in range(0, img_num):
        input_name = list(os.walk(input_folder, topdown=False))[-1][-1][i]
        GT_name = list(os.walk(GT_folder, topdown=False))[-1][-1][i]
        
        input_dir = input_folder+'//'+input_name
        GT_dir = GT_folder+'//'+GT_name
        print('read im_name -----> ',input_dir)
        print('read im_name -----> ',GT_dir)
        input_img = tiff.imread(input_dir)
        GT_img = tiff.imread(GT_dir)
        # im = im.transpose(2,0,1)
        print(input_img.shape)

        # GT_img = np.expand_dims(GT_img, axis=0)
        # input_img = input_img.transpose(2,0,1)
        input_img = input_img.astype(np.float32)/normalize_factor

        # if input_pretype == 'mean':
        #     print('input_pretype == mean')
        #     input_img_ave_single = np.mean(input_img, axis=0)
        #     input_img_ave = np.zeros(input_img.shape)
        #     for i in range(0, input_img.shape[0]):
        #         input_img_ave[i,:,:] = input_img_ave_single
        #     input_img = input_img-input_img_ave

        # input_img = input_img-np.min(input_img)
        # norm_rate = 2000
        # input_img = input_img/np.max(input_img)*norm_rate-norm_rate//2

        input_img = input_img-np.min(input_img)
        norm_rate = 1000
        input_img = input_img/np.max(input_img)*norm_rate
        # img_list[im_name] = noise_im

        if GT_pretype == 'min':
            print('GT_pretype == min')
            GT_img_ave_single = np.min(GT_img, axis=0)
            GT_img_ave = np.zeros(GT_img.shape,dtype='float32')
            for i in range(0, GT_img.shape[0]):
                GT_img_ave[i,:,:] = GT_img_ave_single
            GT_img = GT_img-GT_img_ave

        GT_list[GT_name.replace('.tif','')] = GT_img
        input_list[input_name.replace('.tif','')] = input_img
        name_list.append(GT_name.replace('.tif',''))
    
    num_per_img = math.ceil(train_datasets_size/img_num)
    coor_list = {}
    for i in range(0, img_num):
        for ii in range(0, num_per_img):
            per_coor = {}
            init_w = np.random.randint(0, GT_img.shape[-2]-img_w-1)
            init_h = np.random.randint(0, GT_img.shape[-1]-img_h-1)
            init_s = np.random.randint(0, GT_img.shape[0]-img_s-1)

            per_coor['name'] = name_list[i]
            per_coor['init_w'] = init_w
            per_coor['init_h'] = init_h
            per_coor['init_s'] = init_s

            per_coor['end_w'] = init_w+img_w
            per_coor['end_h'] = init_h+img_h
            per_coor['end_s'] = init_s+img_s
            per_coor['name'] = name_list[i]

            patch_name = name_list[i]+'_x'+str(init_h)+'_y'+str(init_w)+'_z'+str(init_s)
            patch_name_list.append(patch_name.replace('.tif',''))
            coor_list[patch_name] = per_coor
    return  patch_name_list, coor_list, GT_list, input_list


class trainset_RMBG2D(Dataset):
    def __init__(self, name_list, coor_list, GT_list, raw_list):
        self.name_list = name_list
        self.coor_list=coor_list
        self.GT_list = GT_list
        self.raw_list = raw_list

    def __getitem__(self, index):
        #fn = self.images[index]
        patch_name = self.name_list[index]
        single_coordinate = self.coor_list[patch_name]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        name = single_coordinate['name']
        raw = self.raw_list[name]
        GT = self.GT_list[name]
        input = raw[init_s: end_s, init_h:end_h, init_w:end_w]
        # target = GT[init_s: end_s, init_h:end_h, init_w:end_w]
        target = GT[init_s+(end_s-init_s)//2, init_h:end_h, init_w:end_w]
        if target.shape==2:
            target = target[np.newaxis,:,:]

        if_random_imresize = 0
        if_random_transform = 0
        if if_random_transform:
            input = random_transform(input)
            # print('input -----> ',input.shape)

        input=torch.from_numpy(input).cuda().float()
        target=torch.from_numpy(np.expand_dims(target, 0)).cuda().float()
        # print('input -----> ',input.shape)

        if if_random_imresize:
            input = random_imresize(input)
            # print('input -----> ',input.shape)
        #target = self.target[index]
        # print('input -----> ',input.shape)
        # print('target -----> ',target.shape)
        return input, target, patch_name

    def __len__(self):
        return len(self.name_list)