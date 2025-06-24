from deepwonder.test_SR_acc_signal import test_sr_net_acc
from deepwonder.test_RMBG_acc import test_rmbg_net_acc
from deepwonder.test_SEG_acc import test_seg_net_acc
from deepwonder.test_MN import calculate_neuron
from deepwonder.test_DENO_acc import test_DENO_net
from deepwonder.test_TR import adjust_time_resolution, get_data_fingerprint

import datetime
import time
import numpy as np
import torch
import os


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')


    from para_dict import SEG_para, config_SEG_para
    # sftp://zgx@166.111.72.183:9004/mnt/disk/zgx/DeepLonder_p/results_rush3d/SANeuron_patch_1_up3.5_3/STEP_4_RMBG/reg_view_161_patch_1_g_6f.tif

    SEG_para['GPU'] = '0'
    SEG_para['SEG_datasets_path'] = '..//results_rush3d//SANeuron_patch_1_up3.5_3'
    SEG_para['SEG_datasets_folder'] = 'STEP_4_RMBG'
    SEG_para['SEG_output_dir'] = SEG_para['SEG_datasets_path']+'_seg_test'

    if not os.path.exists(SEG_para['SEG_output_dir']): 
        os.mkdir(SEG_para['SEG_output_dir'])

    for i in range(10, 400, 10):
        pth_name = 'seg_'+str(i)
        SEG_para['SEG_output_folder'] = pth_name
        SEG_para['SEG_pth_index'] = pth_name+'.pth'


        SEG_para = config_SEG_para(SEG_para, 
        SEG_para['SEG_datasets_path']+'//'+SEG_para['SEG_datasets_folder'])

        SEG_model_test = test_seg_net_acc(SEG_para)
        SEG_model_test.run()