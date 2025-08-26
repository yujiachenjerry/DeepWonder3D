import tifffile as tiff
import cv2
import numpy as np
from skimage import io

import time
import datetime
import scipy.io as scio

import math
import os
from .merge_neuron_f import Split_Neuron, listAddtrace
from .merge_neuron_f import listAddcontours_Laplacian_pytorch, list2contours
from .merge_neuron_f import group_mask, calculate_trace, Mining_rest_neuron, clear_neuron
from .merge_neuron_f import Neuron_List_Initial, list2mask, listAddtrace, centroid_distance, list_union, \
    initial_mask_list
from .merge_neuron_f import list2contours, Joint_Mask_List_Simple, Joint_Mask_List_Mul
import multiprocessing as mp


def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0 / pixel_number
    # 发现bins必须写到257，否则255这个值只能分到[254,255)区间
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        # print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    # print(final_thresh)
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img, final_thresh


def merge_neuron_SEG_mul_inten(mask_stack,
                               raw_image,
                               quit_round_rate=0.8,
                               good_round_rate=0.9,
                               good_round_size_rate=0.8,
                               corr_mark=0.9,
                               max_value=1000,
                               seg_thres=0.7,
                               if_nmf=True,
                               inten_thres=0.01,
                               edge_value=20,
                               smallest_neuron_area=36):
    # 1um 144
    # 2um 36

    smallest_neuron_area = smallest_neuron_area

    time_start = time.time()

    inten_thres = inten_thres
    raw_image_max = np.max(raw_image, axis=0)
    raw_image_std = np.std(raw_image, axis=0)
    raw_image_mean = np.mean(raw_image, axis=0)

    #######################################################
    # Ostu thresholding
    #######################################################
    '''
    mask_stack = mask_stack - np.min(mask_stack)
    mask_stack = mask_stack/np.max(mask_stack)*255

    mask_stack_max = np.max(mask_stack, axis=0)
    max_value = np.max(mask_stack)
    _, threshold = otsu(mask_stack_max)
    threshold = threshold*0.75 # for chenqian
    print('threshold ---> ',threshold)
    mask_stack[mask_stack>threshold] = max_value
    mask_stack[mask_stack<threshold] = 0

    mask_stack_filted = mask_stack
    # from skimage import io
    # io.imsave('test.tif', mask_stack_filted)
    '''

    mask_stack = mask_stack - np.percentile(mask_stack, 0.1, axis=(1, 2), keepdims=True)
    mask_stack = mask_stack / np.max(mask_stack, axis=(1, 2), keepdims=True)
    mask_stack = mask_stack.clip(0, 1) * 255
    max_value = np.max(mask_stack)
    print('max_value ---> ', max_value)

    N, H, W = mask_stack.shape
    masks = []
    for ii in range(N):
        mask = mask_stack[ii]
        _, threshold = otsu(mask)
        threshold = threshold * 0.7  # 0.7 is adjustable
        print(f'SEG {ii + 1}/{N}: threshold ---> ', threshold)
        mask[mask >= threshold] = max_value
        mask[mask < threshold] = 0
        masks.append(mask)

    masks = np.array(masks)
    mask_stack_filted = masks
    mask_stack = masks

    #######################################################
    w_good_neuron_list = []
    w_bad_neuron_list = []
    g_f_contours = np.zeros(mask_stack.shape)
    b_f_contours = np.zeros(mask_stack.shape)

    time_cost = time.time() - time_start
    # print('[Neuron Segmentation Mask Initialize Time Cost: %.0d s] \n'%(time_cost), end=' ')
    # 是否加速
    if_mul = 1
    if if_mul:
        num_cores = int(mp.cpu_count())
        pool = mp.Pool(num_cores)
        mask_dict = {}
        for i in range(0, mask_stack.shape[0]):
            mask_dict[str(i)] = mask_stack[i, :, :].squeeze()
        results = [
            pool.apply_async(initial_mask_list, args=(mask, quit_round_rate, good_round_rate, smallest_neuron_area)) for
            name, mask in mask_dict.items()]
        for p in results:
            good_neuron_list, bad_neuron_list = p.get()
            w_good_neuron_list = good_neuron_list + w_good_neuron_list
            w_bad_neuron_list = w_bad_neuron_list + bad_neuron_list
            # print('good_neuron_list ---> ',len(good_neuron_list),' bad_neuron_list ---> ',len(bad_neuron_list))
        # print('w_good_neuron_list ---> ',len(w_good_neuron_list),' w_bad_neuron_list ---> ',len(w_bad_neuron_list))
        time_cost = time.time() - time_start
        print('[Neuron Segmentation Mask Initialize Time Cost: %.0d s] \n' % time_cost, end=' ')
        pool.close()
        pool.join()

    # w_n_list_init = w_good_neuron_list #+w_bad_neuron_list
    w_good_neuron_list = listAddtrace(w_good_neuron_list, raw_image)

    # print('w_n_list_init ---> ',len(w_n_list_init))
    # print('w_good_neuron_list ---> ',len(w_good_neuron_list))
    if_time = True
    if if_time:
        time_cost = time.time() - time_start
        print('[1 Time Cost: %.0d s] \n' % time_cost, end=' ')

    if_split = True
    w_good_neuron_list = Joint_Mask_List_Mul(w_good_neuron_list,
                                             corr_mark=0.9,
                                             area_mark=0.9,
                                             active_rate=0,
                                             if_coor=True,
                                             if_area=True,
                                             if_merge=False)
    w_n_list_init = w_good_neuron_list
    w_g_neuron_list = w_good_neuron_list

    time_cost = time.time() - time_start
    print('[2 Time Cost: %.0d s] \n' % (time_cost), end=' ')
    # print('w_g_neuron_list ---> ',len(w_g_neuron_list))
    w_g_neuron_list = Joint_Mask_List_Simple(w_g_neuron_list,
                                             [],
                                             corr_mark=0.7,
                                             area_mark=0.6,
                                             active_rate=0,
                                             if_coor=False,
                                             if_area=True,
                                             if_merge=False)
    # print(w_g_neuron_list)
    time_cost = time.time() - time_start
    print('[3 Time Cost: %.0d s] \n' % time_cost, end=' ')
    # print('w_g_neuron_list ---> ',len(w_g_neuron_list))
    # print('mask -----> ','mask' in w_g_neuron_list[0])
    from .merge_neuron_f import neuron_max_filter, list_add_mask, delete_edge_neuron
    from .merge_neuron_f import remain_mask, add_remain_mask_list, remain_mask_test_list

    all_neuron_mask = list_add_mask(w_g_neuron_list, raw_image)
    time_cost = time.time() - time_start
    print('[4 Time Cost: %.0d s] \n' % time_cost, end=' ')
    # remain_mask part is so slow
    # all_neuron_remain_mask = remain_mask_test_list(w_g_neuron_list)
    # all_neuron_remain_mask = remain_mask(all_neuron_mask) 
    all_neuron_remain_mask = all_neuron_mask
    time_cost = time.time() - time_start
    print('[5 Time Cost: %.0d s] \n' % time_cost, end=' ')
    # w_g_neuron_list1 = w_g_neuron_list #.clone()
    if 0:
        w_g_neuron_list = neuron_max_filter(w_g_neuron_list,
                                            all_neuron_mask,
                                            raw_image_max,
                                            max_thres=inten_thres)
    time_cost = time.time() - time_start
    print('[6 Time Cost: %.0d s] \n' % time_cost, end=' ')
    # print('w_g_neuron_list ---> ',len(w_g_neuron_list))
    # print('mask -----> ','mask' in w_g_neuron_list[0]) 
    w_g_neuron_list = neuron_max_filter(w_g_neuron_list,
                                        all_neuron_mask,
                                        raw_image_std,
                                        max_thres=inten_thres)
    w_g_neuron_list = delete_edge_neuron(w_g_neuron_list,
                                         all_neuron_mask,
                                         edge_value=edge_value)
    time_cost = time.time() - time_start
    print('[7 Time Cost: %.0d s] \n' % time_cost, end=' ')
    # print('w_g_neuron_list ---> ',len(w_g_neuron_list))
    # print('mask -----> ','mask' in w_g_neuron_list[0])
    w_g_neuron_list = add_remain_mask_list(w_g_neuron_list, all_neuron_remain_mask)
    # print('remain_position ---> ',w_g_neuron_list[0]['remain_position'][0,:])
    # print('mask -----> ','mask' in w_g_neuron_list[0])
    time_cost = time.time() - time_start
    print('[8 Time Cost: %.0d s] \n' % time_cost, end=' ')
    return w_g_neuron_list, w_n_list_init, all_neuron_mask, all_neuron_remain_mask, mask_stack_filted
