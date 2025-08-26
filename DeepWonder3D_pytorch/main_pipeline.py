from deepwonder.test_SR_acc_signal import test_sr_net_acc
from deepwonder.test_RMBG_acc import test_rmbg_net_acc
from deepwonder.test_SEG_acc import test_seg_net_acc
from deepwonder.test_MN import calculate_neuron
from deepwonder.test_DENO_acc import test_DENO_net
from deepwonder.test_TR import adjust_time_resolution, get_data_fingerprint

import os
import json
import datetime
from natsort import natsorted
from time import time, sleep
import numpy as np
import torch
import sys
import gc


def clear_large_variables(threshold=1 * 1024 * 1024):  # 默认阈值为1MB
    # 获取所有全局变量的名称和值
    global_vars = globals().copy()  # 使用.copy()以避免在迭代时修改字典
    for var_name, var_value in global_vars.items():
        # 获取变量的大小
        size_in_bytes = sys.getsizeof(var_value)
        # 检查是否大于阈值
        if size_in_bytes > threshold:
            # 删除变量
            del globals()[var_name]
            print(f"Variable '{var_name}' of size {size_in_bytes} bytes has been deleted.")
    # 执行垃圾收集
    gc.collect()
    print("Garbage collection completed.")


def main_pipeline(input_path,
                  input_folder,
                  SR_up_rate,
                  GPU_index,
                  output_dir,
                  t_resolution=10,
                  output_folder='',
                  type='deno_sr_rmbg_seg_mn',
                  denoise_index=0):
    NOW_path = input_path
    NOW_folder = input_folder
    ############ DENO #########################################
    ###########################################################

    DENO_datasets_path = NOW_path
    DENO_datasets_folder = NOW_folder
    DENO_output_dir = output_dir
    DENO_output_folder = 'STEP_1_DENO'

    from para_dict import DENO_para, config_DENO_para

    DENO_para['GPU'] = GPU_index
    DENO_para['DENO_datasets_path'] = DENO_datasets_path
    DENO_para['DENO_datasets_folder'] = DENO_datasets_folder

    DENO_para['DENO_output_dir'] = DENO_output_dir
    DENO_para['DENO_output_folder'] = DENO_output_folder
    DENO_para['denoise_index'] = denoise_index

    if 'deno' in type:
        DENO_para = config_DENO_para(DENO_para,
                                     DENO_datasets_path + '//' + DENO_datasets_folder)

        DENO_model_test = test_DENO_net(DENO_para)

        t_DENO = -time()
        DENO_model_test.run()
        t_DENO += time()

        NOW_path = output_dir
        NOW_folder = DENO_output_folder
        torch.cuda.empty_cache()
        clear_large_variables()
    else:
        print('DENO (denoising) is not in type')
        t_DENO = -1

    ############ TR #########################################
    ###########################################################
    if t_resolution != 10 and ('tr' in type):
        TR_output_dir = output_dir
        TR_output_folder = 'STEP_2_TR'

        t_TR = -time()
        adjust_time_resolution(input_path=NOW_path,
                               input_folder=NOW_folder,
                               output_path=TR_output_dir,
                               output_folder=TR_output_folder,
                               t_resolution=t_resolution)
        t_TR += time()

        NOW_path = output_dir
        NOW_folder = TR_output_folder
        torch.cuda.empty_cache()
        clear_large_variables()
    else:
        print('TR (adjust time resolution) is not in type')
        t_TR = -1
    ############ SR #########################################
    ###########################################################
    SR_output_dir = output_dir
    SR_output_folder = 'STEP_3_SR'

    from para_dict import SR_para, config_SR_para

    SR_para['GPU'] = GPU_index
    SR_para['up_rate'] = SR_up_rate
    SR_para['datasets_folder'] = NOW_folder
    SR_para['datasets_path'] = NOW_path
    SR_para['output_dir'] = SR_output_dir
    SR_para['SR_output_folder'] = SR_output_folder  # 'SR'

    print('SR_para -----> ', SR_para)
    if 'sr' in type:
        SR_para = config_SR_para(SR_para,
                                 SR_para['datasets_path'] + '//' + SR_para['datasets_folder'])

        SR_model_test = test_sr_net_acc(SR_para)

        t_SR = -time()
        SR_model_test.run()
        t_SR += time()

        NOW_path = output_dir
        NOW_folder = SR_output_folder
        torch.cuda.empty_cache()
        clear_large_variables()
    else:
        print('SR (super resolution) is not in type')
        t_SR = -1
    ############ RMBG #########################################
    ###########################################################

    RMBG_output_dir = output_dir
    RMBG_output_folder = 'STEP_4_RMBG'

    from para_dict import RMBG_para, config_RMBG_para

    RMBG_para['GPU'] = GPU_index
    RMBG_para['RMBG_datasets_path'] = NOW_path
    RMBG_para['RMBG_datasets_folder'] = NOW_folder  # +'//STEP_3_SR'
    RMBG_para['RMBG_output_dir'] = RMBG_output_dir
    RMBG_para['RMBG_output_folder'] = RMBG_output_folder

    if 'rmbg' in type:
        RMBG_para = config_RMBG_para(RMBG_para,
                                     RMBG_para['RMBG_datasets_path'] + '//' + RMBG_para['RMBG_datasets_folder'])

        RMBG_para['RMBG_batch_size'] = 1
        RMBG_model_test = test_rmbg_net_acc(RMBG_para)

        t_RMBG = -time()
        RMBG_model_test.run()
        t_RMBG += time()

        NOW_path = output_dir
        NOW_folder = RMBG_output_folder
        torch.cuda.empty_cache()
        clear_large_variables()
    else:
        print('RMBG (remove background) is not in type')
        t_RMBG = -1
    ###################### SEG #############
    ########################################

    SEG_output_dir = output_dir
    SEG_output_folder = 'STEP_5_SEG'

    from para_dict import SEG_para, config_SEG_para

    SEG_para['GPU'] = GPU_index
    SEG_para['SEG_datasets_path'] = NOW_path
    SEG_para['SEG_datasets_folder'] = NOW_folder
    SEG_para['SEG_output_dir'] = SEG_output_dir
    SEG_para['SEG_output_folder'] = SEG_output_folder

    if 'seg' in type:
        SEG_para = config_SEG_para(SEG_para,
                                   SEG_para['SEG_datasets_path'] + '//' + SEG_para['SEG_datasets_folder'])

        SEG_model_test = test_seg_net_acc(SEG_para)

        t_SEG = -time()
        SEG_model_test.run()
        t_SEG += time()

        NOW_path = output_dir
        NOW_folder = SEG_output_folder
        torch.cuda.empty_cache()
        clear_large_variables()
    else:
        print('SEG (segmentation) is not in type')
        t_SEG = -1

    ###################### MN ##############
    ########################################
    MN_output_dir = output_dir
    MN_output_folder = 'STEP_6_MN'

    from para_dict import MN_para

    MN_para['RMBG_datasets_folder'] = RMBG_output_folder
    MN_para['RMBG_datasets_path'] = output_dir

    MN_para['SEG_datasets_path'] = output_dir
    MN_para['SEG_datasets_folder'] = SEG_output_folder

    # print('SR_output_dir ---> ', SR_output_dir, RMBG_output_dir)
    MN_para['SR_datasets_path'] = output_dir
    MN_para['SR_datasets_folder'] = SR_output_folder

    MN_para['MN_output_dir'] = MN_output_dir
    MN_para['MN_output_folder'] = MN_output_folder

    if 'mn' in type:
        MN_model = calculate_neuron(MN_para)

        t_MN = -time()
        MN_model.run()
        t_MN += time()
    else:
        print('MN (merge neurons) is not in type')
        t_MN = -1

    ################ Save running time ##############
    #################################################
    times = {'DENO': t_DENO, 'TR': t_TR, 'SR': t_SR, 'RMBG': t_RMBG, 'SEG': t_SEG, 'MN': t_MN}
    T_output_dir = output_dir
    T_output_folder = os.path.join(T_output_dir, 'times')
    os.makedirs(T_output_folder, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    json_name = f'times_{current_time}.json'
    with open(os.path.join(T_output_folder, json_name), 'w') as f:
        json.dump(times, f, indent=4)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # the neuron size after super resolution should be 15-20 pixel
    now_pixel_size = 4
    target_pixel_size = 2

    GPU_index = '0'
    t_resolution = 4

    SR_up_rate = int(now_pixel_size / target_pixel_size / 1 * 10) / 10
    input_datasets_path = './datasets'
    input_datasets_folder = 'test'
    output_path = './results'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_dir = os.path.join(output_path, 'DW3D_' + input_datasets_folder + '_up' + str(SR_up_rate) + '_index0')
    main_pipeline(input_path=input_datasets_path,
                  input_folder=input_datasets_folder,
                  SR_up_rate=SR_up_rate,
                  GPU_index=GPU_index,
                  output_dir=output_dir,
                  t_resolution=t_resolution,
                  output_folder='',
                  type='deno_tr_sr_rmbg_seg_mn',
                  denoise_index=0)
