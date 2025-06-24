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


SR_para = { 'GPU' : '1',
            'output_dir' : './/test_results',
            ###########################
            'net_type':'trans_mini',  # trans_mini  trans
            'test_datasize' : 10000,
            
            ###########################
            'img_w' : 48,
            'img_h' : 48,
            'img_s' : 5,
            'batch_size' : 24,
            ###########################
            'signal_SR_img_s' : 5,
            'gap_w' : 32,
            'gap_h' : 32,
            'gap_s' : 1,
            ###########################
            'signal_SR_norm_factor' : 1,
            'signal_SR_pth_path' : "pth//SR_pth",
            'datasets_path' : '..//datasets',
            'datasets_folder' : 'test_07192023_2_113', 
            'up_rate' : 12,

            'signal_SR_pth_index' : "signal_SR_1300.pth",  
            # 'signal_SR_model' : "SR_mean_up15_down5_202309201927_k3", 
            'signal_SR_model' : "SR_HALF_mean__up15_down5_202401091538",
            ###########################
            'signal_SR_f_maps' : 16,  # 32 16
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
            'mean_SR_input_pretype' : '',  # mean
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




RMBG_para = { 'GPU' : '0,1',
                'RMBG_output_dir' : './/test_results',
                ###########################
                'RMBG_datasets_folder' : 'pred_signal',
                'RMBG_datasets_path' : '..//Deepwonder4//test_results//SR2_test_07192023_2_up12_309201927_k3_113',
                'RMBG_test_datasize' : 10000,
                ###########################
                'RMBG_img_w' : 128,
                'RMBG_img_h' : 512,
                'RMBG_img_s' : 96,
                'RMBG_batch_size' : 1,
                'RMBG_select_img_num' : 20000,
                ###########################
                'RMBG_gap_w' : 480,
                'RMBG_gap_h' : 480,
                'RMBG_gap_s' : 72,   # RMBG_img_s*0.75
                ###########################
                'RMBG_norm_factor' : 1,
                'RMBG_pth_path' : "pth//RMBG_pth",
                'RMBG_pth_index' : "RMBG_100.pth",   # RMBG_77
                'RMBG_model' : "RMBG_UNet3D_squeeze_202312212020", # "RMBG_UNet3D_squeeze_202309121346",     
                ###########################
                'RMBG_f_maps' : 8, # 32
                'RMBG_in_c' : 1, # 1
                'RMBG_out_c' : 1, #1 
                'RMBG_input_pretype' : '',
}



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


SEG_para = { 'GPU' : '0,1',
                'SEG_output_dir' : './/test_results',
                ###########################
                'SEG_datasets_folder' : 'RMBG_pred_signal_202310231446',
                'SEG_datasets_path' : 'test_results',
                'SEG_test_datasize' : 10000,
                ###########################
                'SEG_img_w' : 192,
                'SEG_img_h' : 512,
                'SEG_img_s' : 64,
                'SEG_batch_size' : 1,
                'SEG_select_img_num' : 20000,
                ###########################
                'SEG_gap_w' : 480,
                'SEG_gap_h' : 480,
                'SEG_gap_s' : 32,
                ###########################
                'SEG_norm_factor' : 1,
                'SEG_pth_path' : "pth//SEG_pth",
                'SEG_pth_index' : "seg_100.pth",   # seg_100
                'SEG_model' : "Unet4_no_end_UNet3D_20231227-2251", #"Unet4_no_end_UNet3D_20231226-2253", # "3DUNetFFD_20231019-1059",    # Unet4_no_end_UNet3D_20231227-2251 
                ###########################
                'SEG_f_maps' : 8,
                'SEG_in_c' : 1, # 4
                'SEG_out_c' : 1, # 4
}




MN_para = { 'RMBG_datasets_folder' : 'RMBG_pred_signal_202310231446',
                'RMBG_datasets_path' : 'test_results',
                ###########################
                'SEG_datasets_folder' : 'SEG_RMBG_pred_signal_202310231446_202310231451',
                'SEG_datasets_path' : 'test_results',
                ###########################
                'SR_datasets_folder' : 'pred_signal',
                'SR_datasets_path' : 'test_results//SR_test_07192023_2_113_up12',
                'MN_output_dir' : 'test_results',
            }