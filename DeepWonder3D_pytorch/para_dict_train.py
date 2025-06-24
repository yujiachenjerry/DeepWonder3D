import numpy as np

def get_data_fingerprint(data_path):
    im_w_list = []
    im_h_list = []
    im_s_list = []
    import os
    for im_name in list(os.walk(data_path, topdown=False))[-1][-1]:
        if '.tif' in im_name:
            import tifffile as tiff
            im_dir = data_path+'//'+im_name
            im = tiff.imread(im_dir)

            im_w = im.shape[2]
            im_h = im.shape[1]
            im_s = im.shape[0]

            im_w_list.append(im_w)
            im_h_list.append(im_h)
            im_s_list.append(im_s)

    min_im_w = min(im_w_list)
    min_im_h = min(im_h_list)
    min_im_s = min(im_s_list)
    return im_w_list, im_h_list, im_s_list, min_im_w, min_im_h, min_im_s


############ DENO #########################################
###########################################################
def config_DENO_para(DENO_para, DENO_path, GPU_M=48):
    im_w_list, im_h_list, im_s_list, \
    min_im_w, min_im_h, min_im_s \
    = get_data_fingerprint(DENO_path)
    # print(min_im_w, min_im_h, min_im_s)
    # fprint(im_w_list, im_h_list, im_s_list)
    DENO_GPU_list= {'32':1959,
                    '48':2073,
                    '64':2231,
                    '96':2391,
                    '128':2909,
                    '160':3531,
                    '192':4645,
                    '224':6087,
                    '256':8011}
    bin_set = 32
    import math
    if min_im_w<=min_im_h:
        DENO_para['DENO_img_w'] = (math.ceil(min_im_w/2/bin_set)+1)*bin_set
    if min_im_w>min_im_h:
        DENO_para['DENO_img_w'] = (math.ceil(min_im_h/2/bin_set)+1)*bin_set
    if DENO_para['DENO_img_w']>256:
        DENO_para['DENO_img_w'] = 256
    
    DENO_para['DENO_batch_size'] = math.floor(GPU_M*1000/DENO_GPU_list[str(DENO_para['DENO_img_w'])])

    DENO_para['DENO_img_h'] = DENO_para['DENO_img_w']
    if DENO_para['DENO_img_w']>=64:
        DENO_para['DENO_img_s'] = DENO_para['DENO_img_w']//2
    if DENO_para['DENO_img_w']<64:
        DENO_para['DENO_img_s'] = DENO_para['DENO_img_w']

    if DENO_para['DENO_img_w']>=bin_set*2:
        DENO_para['DENO_gap_w'] = DENO_para['DENO_img_w']-bin_set
    if DENO_para['DENO_img_w']<bin_set*2:
        DENO_para['DENO_gap_w'] = DENO_para['DENO_img_w']//2

    if DENO_para['DENO_img_h']>=bin_set*2:
        DENO_para['DENO_gap_h'] = DENO_para['DENO_img_h']-bin_set
    if DENO_para['DENO_img_h']<bin_set*2:
        DENO_para['DENO_gap_h'] = DENO_para['DENO_img_h']//2

    if DENO_para['DENO_img_s']>=bin_set*2:
        DENO_para['DENO_gap_s'] = DENO_para['DENO_img_s']-bin_set
    if DENO_para['DENO_img_s']<bin_set*2:
        DENO_para['DENO_gap_s'] = DENO_para['DENO_img_s']//2
    
    return DENO_para



DENO_para = { 'GPU' : '0,1',
                'DENO_output_dir' : './/test_results',
                ###########################
                'DENO_datasets_folder' : 'pred_signal',
                'DENO_datasets_path' : '..//Deepwonder4//test_results//SR2_test_07192023_2_up12_309201927_k3_113',
                'DENO_test_datasize' : 10000,
                ###########################
                'DENO_img_w' : 0,
                'DENO_img_h' : 0,
                'DENO_img_s' : 0,
                'DENO_batch_size' : 1,
                'DENO_select_img_num' : 20000,
                ###########################
                'DENO_gap_w' : 16,
                'DENO_gap_h' : 16,
                'DENO_gap_s' : 16,
                ###########################
                'DENO_norm_factor' : 1,
                'DENO_pth_path' : "pth//DENO_pth",
                'DENO_pth_index' : "E_20_Iter_5000.pth",   
                'DENO_model' : "dongyufan3_202312251537",     
                ###########################
                'DENO_f_maps' : 16,
                'DENO_in_c' : 1,
                'DENO_out_c' : 1,
                'DENO_input_pretype' : '',
}



############ SR #########################################
###########################################################
def config_SR_para(SR_para, SR_path, GPU_M=48):
    im_w_list, im_h_list, im_s_list, \
    min_im_w, min_im_h, min_im_s \
    = get_data_fingerprint(SR_path)
    # print(min_im_w, min_im_h, min_im_s)
    # fprint(im_w_list, im_h_list, im_s_list)
    SR_GPU_list= {}

    min_value = min(min_im_w, min_im_h)
    min_value_up = min_value*SR_para['up_rate']
    assert min_value>16

    bin_set = 32
    import math
    mark_value = (math.ceil(min_value_up/2/bin_set)+1)*bin_set
    if mark_value>320:
        mark_value = 320
    SR_para['img_w'] = mark_value//SR_para['up_rate']
    SR_para['img_h'] = SR_para['img_w']
    SR_para['gap_w'] = SR_para['img_w']-8
    SR_para['gap_h'] = SR_para['img_h']-8
    SR_para['img_s'] = 5
    
    return SR_para


SR_para = { 'task_type':'signal',  # signal mean
            'net_type':'ps',  # ps trans_mini2
            'if_D':0,
            'GPU':'1',
            'SR_n_epochs':2000,
            'SR_batch_size':1,
            ############################
            'SR_sub_img_size':32,
            'SR_img_w':720,
            'SR_img_h':720,
            'SR_img_s':5,
            ############################
            'SR_lr':0.00001,
            'SR_b1':0.5,
            'SR_b2':0.999,
            ############################
            'SR_f_maps':16,
            'SR_in_c':1,
            'SR_out_c':1,
            ############################
            'SR_norm_factor':10,
            'SR_output_dir':'./results',
            'SR_datasets_folder':'NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_train/mov_w_bg',  # _only1
            ############################
            'SR_datasets_path':'..//datasets',
            'SR_pth_path':'pth//SR_pth',
            'SR_select_img_num':10000,
            'SR_train_datasets_size':500,
            ############################
            'SR_use_pretrain':0,
            'SR_pretrain_index':'signal_SR_480.pth',
            'SR_pretrain_model':'SR_HALF_mean__up15_down5_202311131524',
            'SR_pretrain_path':'pth//SR_pth',
            ############################
            'SR_sample_up':15,
            'SR_sample_down':3,
            'SR_input_pretype':'mean' ,
}



############ RMBG #########################################
###########################################################
def config_RMBG_para(RMBG_para, RMBG_path, GPU_M=48):
    im_w_list, im_h_list, im_s_list, \
    min_im_w, min_im_h, min_im_s \
    = get_data_fingerprint(RMBG_path)
    # print(min_im_w, min_im_h, min_im_s)
    # fprint(im_w_list, im_h_list, im_s_list)
    SR_GPU_list= {'128':2375,
                    '256':4893,
                    '320':7639,
                    '384':11525,
                    '448':16855,
                    '512':23855}
    RMBG_para['RMBG_img_w'] = 256
    import math
    RMBG_para['RMBG_batch_size'] = math.floor(GPU_M*1000/SR_GPU_list[str(RMBG_para['RMBG_img_w'])])
    min_value = min(min_im_w, min_im_h)
    RMBG_para['RMBG_img_h'] = RMBG_para['RMBG_img_w']
    if RMBG_para['RMBG_img_w']>=128:
        RMBG_para['RMBG_img_s'] = RMBG_para['RMBG_img_w']//2
    if RMBG_para['RMBG_img_w']<128:
        RMBG_para['RMBG_img_s'] = RMBG_para['RMBG_img_w']

    bin_set = 32
    if RMBG_para['RMBG_img_w']>=bin_set*2:
        RMBG_para['RMBG_gap_w'] = RMBG_para['RMBG_img_w']-bin_set
    if RMBG_para['RMBG_img_w']<bin_set*2:
        RMBG_para['RMBG_gap_w'] = RMBG_para['RMBG_img_w']//2

    if RMBG_para['RMBG_img_h']>=bin_set*2:
        RMBG_para['RMBG_gap_h'] = RMBG_para['RMBG_img_h']-bin_set
    if RMBG_para['RMBG_img_h']<bin_set*2:
        RMBG_para['RMBG_gap_h'] = RMBG_para['RMBG_img_h']//2

    if RMBG_para['RMBG_img_s']>=bin_set*2:
        RMBG_para['RMBG_gap_s'] = RMBG_para['RMBG_img_s']-bin_set
    if RMBG_para['RMBG_img_s']<bin_set*2:
        RMBG_para['RMBG_gap_s'] = RMBG_para['RMBG_img_s']//2
    return RMBG_para




RMBG_para={'GPU':'1',
                    'RMBG_n_epochs':300,
                    'RMBG_batch_size':1,
                    ############################
                    'RMBG_img_w':256,
                    'RMBG_img_h':256,
                    'RMBG_img_s':256,
                    ############################
                    'RMBG_lr':0.00005,
                    'RMBG_b1':0.5,
                    'RMBG_b2':0.999,
                    'RMBG_norm_factor':1,
                    'RMBG_f_maps':8,
                    'RMBG_in_c':1,
                    'RMBG_out_c':1,
                    'RMBG_down_num':4,
                    ############################
                    'RMBG_output_dir':'./results',
                    'RMBG_datasets_folder':'NA_0.03_depthrange_200_n_1.00_res_0.8_expanded_soma_1.2_train_only1',
                    'RMBG_GT_folder':'mov_wo_bg',
                    'RMBG_input_folder':'mov_w_bg',
                    'RMBG_datasets_path':'..//datasets',
                    ############################
                    'RMBG_use_pretrain':0,
                    'RMBG_pretrain_index':'RMBG_299.pth',
                    'RMBG_pretrain_model':'RMBG_UNet3D_squeeze_202309121346',
                    'RMBG_pretrain_path':'pth//RMBG_pth',
                    ############################
                    'RMBG_pth_path':'pth//RMBG_pth',
                    'RMBG_select_img_num':10000,
                    'RMBG_train_datasets_size':2000,
                    ############################
                    'RMBG_input_pretype':'mean',  #'mean' # guassian
                    'RMBG_GT_pretype':'min'}


############ SEG #########################################
###########################################################
def config_SEG_para(SEG_para, SEG_path, GPU_M=48):
    im_w_list, im_h_list, im_s_list, \
    min_im_w, min_im_h, min_im_s \
    = get_data_fingerprint(SEG_path)
    # print(min_im_w, min_im_h, min_im_s)
    # fprint(im_w_list, im_h_list, im_s_list)
    SEG_GPU_list= { '128':2585,
                    '192':2977,
                    '256':3689,
                    '320':4581,
                    '384':5645,
                    '448':6891,
                    '512':8307,}

    SEG_para['SEG_img_w'] = 256
    import math
    SEG_para['SEG_batch_size'] = math.floor(GPU_M*1000/SEG_GPU_list[str(SEG_para['SEG_img_w'])])

    min_value = min(min_im_w, min_im_h)
    SEG_para['SEG_img_h'] = SEG_para['SEG_img_w']
    bin_set = 32
    SEG_para['SEG_gap_h'] = SEG_para['SEG_img_h']-bin_set
    SEG_para['SEG_gap_w'] = SEG_para['SEG_img_w']-bin_set
    return SEG_para


SEG_para={'GPU':'0,1',
                'SEG_n_epochs':100,
                'SEG_batch_size':1,
                ############################
                'SEG_img_w':512,
                'SEG_img_h':512,
                'SEG_img_s':64,
                ############################
                'SEG_lr':0.00005,
                'SEG_b1':0.5,
                'SEG_b2':0.999,
                'SEG_norm_factor':1,
                ############################
                'SEG_f_maps':16,
                'SEG_in_c':4,
                'SEG_out_c':4,
                'SEG_down_num':4,
                ############################
                'SEG_output_dir':'./results',
                'SEG_datasets_folder':'seg_circle_mov_wo_bg_0.03_64_300_10_2',
                'SEG_GT_folder':'mask',
                'SEG_input_folder':'image',
                'SEG_datasets_path':'..//datasets',
                ############################
                'SEG_use_pretrain':0,
                'SEG_pretrain_index':'signal_SR_220.pth',
                'SEG_pretrain_model':'SR_signal_up15_down5_20230302-2250',
                'SEG_pretrain_path':'pth//SEG_pth',
                ############################
                'SEG_pth_path':'pth//SEG_pth',
                'SEG_select_img_num':10000,
                'SEG_train_datasets_size':2000}



