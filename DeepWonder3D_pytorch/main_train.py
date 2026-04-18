from deepwonder.train_SR_acc import train_sr_net_acc
from deepwonder.train_RMBG_acc import train_rmbg_net_acc
from deepwonder.train_SEG_acc import train_seg_net_acc


import datetime
import time
import numpy as np
import torch


def main_train_pipeline(
    input_path,
    input_folder,
    SR_up_rate,
    GPU_index,
    output_dir,
    t_resolution=10,
    output_folder='',
    type='sr',
):
    """
    Run the main training pipeline for one or more deep learning models.

    This function configures and launches training for super-resolution (SR),
    background removal (RMBG), and/or segmentation (SEG) networks depending on
    the content of the ``type`` argument.

    Args:
        input_path (str): Base directory that contains the training datasets.
            Expected format is a filesystem path, for example ``'./data'`` or
            ``'/path/to/input'``.
        input_folder (str): Name of the subfolder inside ``input_path`` that
            stores the training data for the selected models.
        SR_up_rate (int): Upsampling factor used by the SR model, typically a
            positive integer such as 2, 4, or 8.
        GPU_index (str): Comma-separated list of GPU indices to use, for
            example ``'0'`` or ``'0,1'``.
        output_dir (str): Base output directory where training logs, checkpoints,
            and final model weights will be written.
        t_resolution (int, optional): Time resolution in milliseconds reserved
            for potential temporal settings. Currently unused but kept for
            forward compatibility.
        output_folder (str, optional): Name of the subfolder under
            ``output_dir`` to store results. If empty, a default step folder
            name will be used.
        type (str, optional): Model type selector string. If it contains
            ``'sr'``, the SR model is trained; if it contains ``'rmbg'``,
            the RMBG model is trained; if it contains ``'seg'``, the
            segmentation model is trained. Multiple keywords can be combined.

    Returns:
        None: All artifacts are saved to disk under the specified
        ``output_dir`` and related subfolders.

    Notes:
        - Each model uses its own configuration dictionary imported from
          ``para_dict_train``.
        - Training objects are instantiated by helper functions such as
          ``train_sr_net_acc`` and then executed via their ``run`` method.
        - Multiple model types can be trained sequentially in a single call,
          depending on the value of ``type``.
    """
    NOW_path = input_path
    NOW_folder = input_folder

    if 'sr' in type:
        ############ SR #########################################
        ###########################################################
        SR_output_dir = output_dir
        SR_output_folder = 'STEP_3_SR'

        from para_dict_train import SR_para, config_SR_para
        
        SR_para['GPU'] = GPU_index
        if NOW_folder!='':
            SR_para['datasets_folder'] =  NOW_folder
        if NOW_path!='':
            SR_para['datasets_path'] =  NOW_path
        if SR_output_dir!='':
            SR_para['output_dir'] = SR_output_dir
        if SR_output_folder!='':
            SR_para['SR_output_folder'] = SR_output_folder
        print('SR_para -----> ',SR_para)

        SR_model_train = train_sr_net_acc(SR_para)
        SR_model_train.run()

    if 'rmbg' in type:
        ############ RMBG #########################################
        ###########################################################
        RMBG_output_dir = output_dir
        RMBG_output_folder = 'STEP_4_RMBG'

        from para_dict_train import RMBG_para, config_RMBG_para

        RMBG_para['GPU'] = GPU_index
        if NOW_folder!='':
            RMBG_para['RMBG_datasets_path'] = NOW_path
        if NOW_folder!='':
            RMBG_para['RMBG_datasets_folder'] = NOW_folder
        if RMBG_output_dir!='':
            RMBG_para['RMBG_output_dir'] = RMBG_output_dir
        if RMBG_output_folder!='':
            RMBG_para['RMBG_output_folder'] = RMBG_output_folder
        print('RMBG_para -----> ',RMBG_para)

        # SR_model_train = train_sr_net_acc(SR_para)
        # SR_model_train.run()
        RMBG_model_train = train_rmbg_net_acc(RMBG_para)
        RMBG_model_train.run()

    if 'seg' in type:
        ############ RMBG #########################################
        ###########################################################
        SEG_output_dir = output_dir
        SEG_output_folder = 'STEP_5_SEG'

        from para_dict_train import SEG_para

        SEG_para['GPU'] = GPU_index
        if NOW_folder!='':
            SEG_para['SEG_datasets_path'] = NOW_path
        if NOW_folder!='':
            SEG_para['SEG_datasets_folder'] = NOW_folder
        if SEG_output_dir!='':
            SEG_para['SEG_output_dir'] = SEG_output_dir
        if SEG_output_folder!='':
            SEG_para['SEG_output_folder'] = SEG_output_folder
        print('SEG_para -----> ',SEG_para)

        # SR_model_train = train_sr_net_acc(SR_para)
        # SR_model_train.run()
        SEG_model_train = train_seg_net_acc(SEG_para)
        SEG_model_train.run()





if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    pixel_size = 6
    GPU_index  = '0'
    t_resolution = 50

    SR_up_rate = pixel_size
    input_datasets_path = ''
    input_datasets_folder = ''
    output_dir = ''

    main_train_pipeline(input_path = input_datasets_path, 
                    input_folder = input_datasets_folder, 
                    SR_up_rate = SR_up_rate, 
                    GPU_index = GPU_index, 
                    output_dir = output_dir, 
                    t_resolution = t_resolution ,
                    output_folder = '', 
                    type = 'sr')