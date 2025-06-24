import numpy as np

DENO_para['GPU'] = GPU_index
DENO_para['DENO_datasets_path'] = DENO_datasets_path
DENO_para['DENO_datasets_folder'] = DENO_datasets_folder

DENO_para['DENO_output_dir'] = DENO_output_dir
DENO_para['DENO_output_folder'] = DENO_output_folder


DENO_model_test = test_DENO_net_acc(DENO_para)


# DENO 128 128 16 GPU 2165
# DENO 128 128 32 GPU 2411
# DENO 128 128 48 GPU 2669
# DENO 128 128 64 GPU 2909
# DENO 128 128 80 GPU 3035
# DENO 128 128 96 GPU 3279

# float 32
# DENO 32 32 32 GPU 1959
# DENO 48 48 48 GPU 2073
# DENO 64 64 64 GPU 2231
# DENO 96 96 48 GPU 2391
# DENO 128 128 64 GPU 2909
# DENO 160 160 80 GPU 3531
# DENO 192 192 96 GPU 4645
# DENO 224 224 112 GPU 6087
# DENO 256 256 128 GPU 8011

# DENO 80 80 40 GPU 2223
# DENO 112 112 56 GPU 2391
# DENO 144 144 72 GPU 3145


# float 32
# DENO 32 32 32 GPU 1959
# DENO 48 48 48 GPU 2073
# DENO 64 64 64 GPU 2231
# DENO 96 96 48 GPU 2391
# DENO 128 128 64 GPU 2909
# DENO 160 160 80 GPU 3531
# DENO 192 192 96 GPU 4645
# DENO 224 224 112 GPU 6087
# DENO 256 256 128 GPU 8011

# DENO 80 80 40 GPU 2223
# DENO 112 112 56 GPU 2391
# DENO 144 144 72 GPU 3145



# SR 48 6 batch=32 23206.44
# SR 48 6 batch=24 17184.44
# SR 48 6 batch=16 12880.44
# SR 48 6 batch=8 7658.44
# SR 48 6 batch=4 5122.44
# SR 48 6 batch=2 3834.44
# SR 48 6 batch=1 3832.44
# SR 48 4 3718.44
# SR 48 1 3638.44


# RMBG 256 4893.44
# RMBG 320 7639.44
# RMBG 384 11525.44
# RMBG 448 16855.44
# RMBG 512 23855.44


# SEG 128 2585.44
# SEG 192 2977.44
# SEG 256 3689.44
# SEG 320 4581.44
# SEG 384 5645.44
# SEG 448 6891.44
# SEG 512 8307


# SEG 128 2585.44
# SEG 192 2977.44
# SEG 256 3689.44
# SEG 320 4581.44
# SEG 384 5645.44
# SEG 448 6891.44
# SEG 512 8307


# RMBG 128 2585.44
# RMBG 192 2585.44
# RMBG 256 4893.44
# RMBG 320 7639.44
# RMBG 384 11525.44
# RMBG 448 16855.44
# RMBG 512 23855.44



model_para_name ----->  Generator.encoders.0.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.encoders.0.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.encoders.1.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.encoders.1.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.encoders.2.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.encoders.2.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.encoders.3.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.encoders.3.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.decoders.0.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.decoders.0.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.decoders.1.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.decoders.1.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.decoders.2.basic_module.SingleConv1.conv.weight
model_para_name ----->  Generator.decoders.2.basic_module.SingleConv2.conv.weight
model_para_name ----->  Generator.final_conv.weight
model_para_name ----->  Generator.final_conv.bias









im_name_list ----->  ['reg_view_115_patch_2_g_27.tiff', 'reg_view_118_patch_2_g_29.tiff', 'reg_view_143_patch_2_g_5.tiff', 'reg_view_71_patch_2_g_25.tiff', 'reg_view_65_patch_2_g_23.tiff', 'reg_view_108_patch_2_g_8.tiff', 'reg_view_155_patch_2_g_14.tiff', 'reg_view_108_patch_2_g_0.tiff', 'reg_view_188_patch_2_g_2.tiff', 'reg_view_108_patch_2_g_23.tiff', 'reg_view_161_patch_2_g_5.tiff', 'reg_view_108_patch_2_g_5.tiff', 'reg_view_108_patch_2_g_1.tiff', 'reg_view_111_patch_2_g_0.tiff', 'reg_view_111_patch_2_g_21.tiff', 'reg_view_155_patch_2_g_1.tiff', 'reg_view_143_patch_2_g_26.tiff', 'reg_view_155_patch_2_g_19.tiff', 'reg_view_188_patch_2_g_15.tiff', 'reg_view_83_patch_2_g_20.tiff', 'reg_view_115_patch_2_g_4.tiff', 'reg_view_115_patch_2_g_19.tiff', 'reg_view_65_patch_2_g_25.tiff', 'reg_view_38_patch_2_g_11.tiff', 'reg_view_155_patch_2_g_3.tiff', 'reg_view_111_patch_2_g_11.tiff', 'reg_view_38_patch_2_g_25.tiff', 'reg_view_108_patch_2_g_15.tiff', 'reg_view_188_patch_2_g_20.tiff', 'reg_view_38_patch_2_g_19.tiff', 'reg_view_65_patch_2_g_3.tiff', 'reg_view_143_patch_2_g_9.tiff', 'reg_view_38_patch_2_g_9.tiff', 'reg_view_108_patch_2_g_29.tiff', 'reg_view_83_patch_2_g_4.tiff', 'reg_view_65_patch_2_g_21.tiff', 'reg_view_161_patch_2_g_10.tiff', 'reg_view_65_patch_2_g_11.tiff', 'reg_view_161_patch_2_g_29.tiff', 'reg_view_161_patch_2_g_12.tiff', 'reg_view_161_patch_2_g_8.tiff', 'reg_view_143_patch_2_g_7.tiff', 'reg_view_71_patch_2_g_26.tiff', 'reg_view_111_patch_2_g_22.tiff', 'reg_view_161_patch_2_g_4.tiff', 'reg_view_113_patch_2_g_0.tiff', 'reg_view_118_patch_2_g_14.tiff', 'reg_view_65_patch_2_g_14.tiff', 'reg_view_71_patch_2_g_1.tiff', 'reg_view_161_patch_2_g_21.tiff', 'reg_view_65_patch_2_g_13.tiff', 'reg_view_71_patch_2_g_14.tiff', 'reg_view_38_patch_2_g_21.tiff', 'reg_view_65_patch_2_g_12.tiff', 'reg_view_113_patch_2_g_24.tiff', 'reg_view_71_patch_2_g_23.tiff', 'reg_view_71_patch_2_g_9.tiff', 'reg_view_161_patch_2_g_2.tiff', 'reg_view_113_patch_2_g_19.tiff', 'reg_view_83_patch_2_g_2.tiff', 'reg_view_108_patch_2_g_7.tiff', 'reg_view_115_patch_2_g_25.tiff', 'reg_view_143_patch_2_g_4.tiff', 'reg_view_188_patch_2_g_29.tiff', 'reg_view_118_patch_2_g_11.tiff', 'reg_view_115_patch_2_g_23.tiff', 'reg_view_115_patch_2_g_14.tiff', 'reg_view_65_patch_2_g_5.tiff', 'reg_view_155_patch_2_g_23.tiff', 'reg_view_65_patch_2_g_17.tiff', 'reg_view_143_patch_2_g_21.tiff', 'reg_view_118_patch_2_g_18.tiff', 'reg_view_113_patch_2_g_2.tiff', 'reg_view_155_patch_2_g_2.tiff', 'reg_view_65_patch_2_g_16.tiff', 'reg_view_65_patch_2_g_7.tiff', 'reg_view_118_patch_2_g_20.tiff', 'reg_view_155_patch_2_g_10.tiff', 'reg_view_111_patch_2_g_15.tiff', 'reg_view_38_patch_2_g_26.tiff', 'reg_view_188_patch_2_g_8.tiff', 'reg_view_155_patch_2_g_27.tiff', 'reg_view_118_patch_2_g_15.tiff', 'reg_view_111_patch_2_g_27.tiff', 'reg_view_108_patch_2_g_2.tiff', 'reg_view_83_patch_2_g_27.tiff', 'reg_view_83_patch_2_g_1.tiff', 'reg_view_108_patch_2_g_10.tiff', 'reg_view_113_patch_2_g_29.tiff', 'reg_view_65_patch_2_g_27.tiff', 'reg_view_83_patch_2_g_11.tiff', 'reg_view_118_patch_2_g_23.tiff', 'reg_view_115_patch_2_g_0.tiff', 'reg_view_115_patch_2_g_20.tiff', 'reg_view_108_patch_2_g_16.tiff', 'reg_view_155_patch_2_g_13.tiff', 'reg_view_71_patch_2_g_3.tiff', 'reg_view_38_patch_2_g_27.tiff', 'reg_view_161_patch_2_g_13.tiff', 'reg_view_188_patch_2_g_13.tiff', 'reg_view_65_patch_2_g_9.tiff', 'reg_view_65_patch_2_g_1.tiff', 'reg_view_115_patch_2_g_8.tiff', 'reg_view_65_patch_2_g_29.tiff', 'reg_view_108_patch_2_g_13.tiff', 'reg_view_161_patch_2_g_23.tiff', 'reg_view_113_patch_2_g_26.tiff', 'reg_view_83_patch_2_g_10.tiff', 'reg_view_108_patch_2_g_27.tiff', 'reg_view_143_patch_2_g_23.tiff', 'reg_view_38_patch_2_g_20.tiff', 'reg_view_118_patch_2_g_0.tiff', 'reg_view_38_patch_2_g_14.tiff', 'reg_view_108_patch_2_g_14.tiff', 'reg_view_188_patch_2_g_21.tiff', 'reg_view_115_patch_2_g_3.tiff', 'reg_view_113_patch_2_g_4.tiff', 'reg_view_71_patch_2_g_17.tiff', 'reg_view_118_patch_2_g_19.tiff', 'reg_view_65_patch_2_g_18.tiff', 'reg_view_143_patch_2_g_27.tiff', 'reg_view_118_patch_2_g_3.tiff', 'reg_view_143_patch_2_g_19.tiff', 'reg_view_143_patch_2_g_2.tiff', 'reg_view_161_patch_2_g_28.tiff', 'reg_view_83_patch_2_g_25.tiff', 'reg_view_113_patch_2_g_15.tiff', 'reg_view_161_patch_2_g_19.tiff', 'reg_view_108_patch_2_g_22.tiff', 'reg_view_143_patch_2_g_11.tiff', 'reg_view_111_patch_2_g_18.tiff', 'reg_view_113_patch_2_g_18.tiff', 'reg_view_143_patch_2_g_8.tiff', 'reg_view_143_patch_2_g_28.tiff', 'reg_view_118_patch_2_g_22.tiff', 'reg_view_113_patch_2_g_10.tiff', 'reg_view_155_patch_2_g_9.tiff', 'reg_view_118_patch_2_g_27.tiff', 'reg_view_143_patch_2_g_16.tiff', 'reg_view_108_patch_2_g_6.tiff', 'reg_view_118_patch_2_g_28.tiff', 'reg_view_143_patch_2_g_18.tiff', 'reg_view_161_patch_2_g_25.tiff', 'reg_view_111_patch_2_g_7.tiff', 'reg_view_143_patch_2_g_14.tiff', 'reg_view_161_patch_2_g_24.tiff', 'reg_view_155_patch_2_g_24.tiff', 'reg_view_118_patch_2_g_8.tiff', 'reg_view_113_patch_2_g_7.tiff', 'reg_view_111_patch_2_g_19.tiff', 'reg_view_188_patch_2_g_4.tiff', 'reg_view_83_patch_2_g_22.tiff', 'reg_view_111_patch_2_g_3.tiff', 'reg_view_161_patch_2_g_27.tiff', 'reg_view_83_patch_2_g_0.tiff', 'reg_view_83_patch_2_g_24.tiff', 'reg_view_83_patch_2_g_8.tiff', 'reg_view_115_patch_2_g_7.tiff', 'reg_view_83_patch_2_g_17.tiff', 'reg_view_83_patch_2_g_15.tiff', 'reg_view_71_patch_2_g_16.tiff', 'reg_view_113_patch_2_g_6.tiff', 'reg_view_155_patch_2_g_11.tiff', 'reg_view_188_patch_2_g_5.tiff', 'reg_view_113_patch_2_g_17.tiff', 'reg_view_188_patch_2_g_18.tiff', 'reg_view_38_patch_2_g_4.tiff', 'reg_view_108_patch_2_g_4.tiff', 'reg_view_161_patch_2_g_6.tiff', 'reg_view_161_patch_2_g_17.tiff', 'reg_view_155_patch_2_g_17.tiff', 'reg_view_111_patch_2_g_8.tiff', 'reg_view_143_patch_2_g_1.tiff', 'reg_view_188_patch_2_g_19.tiff', 'reg_view_115_patch_2_g_26.tiff', 'reg_view_143_patch_2_g_25.tiff', 'reg_view_161_patch_2_g_15.tiff', 'reg_view_83_patch_2_g_7.tiff', 'reg_view_65_patch_2_g_15.tiff', 'reg_view_71_patch_2_g_28.tiff', 'reg_view_188_patch_2_g_24.tiff', 'reg_view_143_patch_2_g_12.tiff', 'reg_view_113_patch_2_g_28.tiff', 'reg_view_83_patch_2_g_29.tiff', 'reg_view_115_patch_2_g_15.tiff', 'reg_view_111_patch_2_g_13.tiff', 'reg_view_118_patch_2_g_7.tiff', 'reg_view_115_patch_2_g_29.tiff', 'reg_view_161_patch_2_g_7.tiff', 'reg_view_111_patch_2_g_10.tiff', 'reg_view_65_patch_2_g_6.tiff', 'reg_view_118_patch_2_g_13.tiff', 'reg_view_118_patch_2_g_21.tiff', 'reg_view_71_patch_2_g_6.tiff', 'reg_view_83_patch_2_g_18.tiff', 'reg_view_38_patch_2_g_17.tiff', 'reg_view_108_patch_2_g_9.tiff', 'reg_view_111_patch_2_g_5.tiff', 'reg_view_38_patch_2_g_16.tiff', 'reg_view_111_patch_2_g_25.tiff', 'reg_view_143_patch_2_g_3.tiff', 'reg_view_155_patch_2_g_26.tiff', 'reg_view_38_patch_2_g_7.tiff', 'reg_view_71_patch_2_g_10.tiff', 'reg_view_71_patch_2_g_2.tiff', 'reg_view_143_patch_2_g_24.tiff', 'reg_view_83_patch_2_g_28.tiff', 'reg_view_111_patch_2_g_1.tiff', 'reg_view_65_patch_2_g_20.tiff', 'reg_view_161_patch_2_g_18.tiff', 'reg_view_38_patch_2_g_6.tiff', 'reg_view_38_patch_2_g_24.tiff', 'reg_view_155_patch_2_g_7.tiff', 'reg_view_143_patch_2_g_6.tiff', 'reg_view_118_patch_2_g_5.tiff', 'reg_view_143_patch_2_g_15.tiff', 'reg_view_83_patch_2_g_5.tiff', 'reg_view_83_patch_2_g_26.tiff', 'reg_view_143_patch_2_g_10.tiff', 'reg_view_143_patch_2_g_29.tiff', 'reg_view_115_patch_2_g_9.tiff', 'reg_view_111_patch_2_g_23.tiff', 'reg_view_113_patch_2_g_14.tiff', 'reg_view_111_patch_2_g_24.tiff', 'reg_view_65_patch_2_g_8.tiff', 'reg_view_65_patch_2_g_19.tiff', 'reg_view_65_patch_2_g_4.tiff', 'reg_view_71_patch_2_g_15.tiff', 'reg_view_115_patch_2_g_13.tiff', 'reg_view_188_patch_2_g_25.tiff', 'reg_view_143_patch_2_g_20.tiff', 'reg_view_38_patch_2_g_13.tiff', 'reg_view_188_patch_2_g_16.tiff', 'reg_view_188_patch_2_g_7.tiff', 'reg_view_118_patch_2_g_26.tiff', 'reg_view_161_patch_2_g_14.tiff', 'reg_view_65_patch_2_g_26.tiff', 'reg_view_188_patch_2_g_22.tiff', 'reg_view_115_patch_2_g_5.tiff', 'reg_view_83_patch_2_g_23.tiff', 'reg_view_155_patch_2_g_12.tiff', 'reg_view_108_patch_2_g_11.tiff', 'reg_view_115_patch_2_g_10.tiff', 'reg_view_113_patch_2_g_9.tiff', 'reg_view_38_patch_2_g_22.tiff', 'reg_view_115_patch_2_g_18.tiff', 'reg_view_113_patch_2_g_5.tiff', 'reg_view_188_patch_2_g_0.tiff', 'reg_view_188_patch_2_g_10.tiff', 'reg_view_161_patch_2_g_16.tiff', 'reg_view_188_patch_2_g_27.tiff', 'reg_view_71_patch_2_g_22.tiff', 'reg_view_71_patch_2_g_0.tiff', 'reg_view_115_patch_2_g_28.tiff', 'reg_view_113_patch_2_g_22.tiff', 'reg_view_108_patch_2_g_21.tiff', 'reg_view_118_patch_2_g_1.tiff', 'reg_view_71_patch_2_g_5.tiff', 'reg_view_38_patch_2_g_23.tiff', 'reg_view_161_patch_2_g_20.tiff', 'reg_view_161_patch_2_g_0.tiff', 'reg_view_188_patch_2_g_28.tiff', 'reg_view_143_patch_2_g_17.tiff', 'reg_view_83_patch_2_g_12.tiff', 'reg_view_71_patch_2_g_21.tiff', 'reg_view_118_patch_2_g_2.tiff', 'reg_view_155_patch_2_g_29.tiff', 'reg_view_118_patch_2_g_17.tiff', 'reg_view_115_patch_2_g_21.tiff', 'reg_view_113_patch_2_g_11.tiff', 'reg_view_108_patch_2_g_28.tiff', 'reg_view_65_patch_2_g_28.tiff', 'reg_view_65_patch_2_g_10.tiff', 'reg_view_111_patch_2_g_4.tiff', 'reg_view_113_patch_2_g_21.tiff', 'reg_view_71_patch_2_g_7.tiff', 'reg_view_38_patch_2_g_1.tiff', 'reg_view_155_patch_2_g_18.tiff', 'reg_view_188_patch_2_g_1.tiff', 'reg_view_71_patch_2_g_20.tiff', 'reg_view_111_patch_2_g_9.tiff', 'reg_view_83_patch_2_g_13.tiff', 'reg_view_65_patch_2_g_22.tiff', 'reg_view_143_patch_2_g_13.tiff', 'reg_view_71_patch_2_g_8.tiff', 'reg_view_113_patch_2_g_12.tiff', 'reg_view_83_patch_2_g_3.tiff', 'reg_view_115_patch_2_g_22.tiff', 'reg_view_113_patch_2_g_8.tiff', 'reg_view_155_patch_2_g_5.tiff', 'reg_view_155_patch_2_g_6.tiff', 'reg_view_38_patch_2_g_10.tiff', 'reg_view_188_patch_2_g_14.tiff', 'reg_view_188_patch_2_g_6.tiff', 'reg_view_188_patch_2_g_17.tiff', 'reg_view_108_patch_2_g_26.tiff', 'reg_view_188_patch_2_g_12.tiff', 'reg_view_108_patch_2_g_24.tiff', 'reg_view_71_patch_2_g_13.tiff', 'reg_view_188_patch_2_g_11.tiff', 'reg_view_108_patch_2_g_25.tiff', 'reg_view_108_patch_2_g_3.tiff', 'reg_view_118_patch_2_g_25.tiff', 'reg_view_143_patch_2_g_0.tiff', 'reg_view_65_patch_2_g_2.tiff', 'reg_view_38_patch_2_g_15.tiff', 'reg_view_108_patch_2_g_17.tiff', 'reg_view_118_patch_2_g_10.tiff', 'reg_view_83_patch_2_g_9.tiff', 'reg_view_115_patch_2_g_12.tiff', 'shifts_wholeprocess_.mat', 'reg_view_155_patch_2_g_28.tiff', 'reg_view_115_patch_2_g_11.tiff', 'reg_view_38_patch_2_g_5.tiff', 'reg_view_113_patch_2_g_13.tiff', 'reg_view_111_patch_2_g_26.tiff', 'reg_view_111_patch_2_g_17.tiff', 'reg_view_38_patch_2_g_3.tiff', 'reg_view_71_patch_2_g_27.tiff', 'reg_view_118_patch_2_g_9.tiff', 'reg_view_161_patch_2_g_3.tiff', 'reg_view_155_patch_2_g_4.tiff', 'reg_view_71_patch_2_g_4.tiff', 'reg_view_115_patch_2_g_17.tiff', 'reg_view_113_patch_2_g_23.tiff', 'reg_view_108_patch_2_g_18.tiff', 'reg_view_155_patch_2_g_8.tiff', 'reg_view_83_patch_2_g_14.tiff', 'reg_view_188_patch_2_g_26.tiff', 'reg_view_113_patch_2_g_27.tiff', 'reg_view_38_patch_2_g_29.tiff', 'reg_view_83_patch_2_g_21.tiff', 'reg_view_118_patch_2_g_12.tiff', 'reg_view_161_patch_2_g_1.tiff', 'reg_view_111_patch_2_g_14.tiff', 'reg_view_71_patch_2_g_19.tiff', 'reg_view_71_patch_2_g_18.tiff', 'reg_view_143_patch_2_g_22.tiff', 'reg_view_188_patch_2_g_23.tiff', 'reg_view_38_patch_2_g_28.tiff', 'reg_view_115_patch_2_g_24.tiff', 'reg_view_71_patch_2_g_12.tiff', 'reg_view_71_patch_2_g_11.tiff', 'reg_view_118_patch_2_g_4.tiff', 'reg_view_111_patch_2_g_28.tiff', 'reg_view_108_patch_2_g_20.tiff', 'reg_view_155_patch_2_g_15.tiff', 'reg_view_188_patch_2_g_9.tiff', 'reg_view_115_patch_2_g_1.tiff', 'reg_view_111_patch_2_g_20.tiff', 'reg_view_155_patch_2_g_16.tiff', 'reg_view_155_patch_2_g_25.tiff', 'reg_view_161_patch_2_g_22.tiff', 'reg_view_188_patch_2_g_3.tiff', 'reg_view_111_patch_2_g_16.tiff', 'reg_view_83_patch_2_g_6.tiff', 'reg_view_155_patch_2_g_20.tiff', 'reg_view_108_patch_2_g_12.tiff', 'reg_view_155_patch_2_g_0.tiff', 'reg_view_111_patch_2_g_29.tiff', 'reg_view_155_patch_2_g_21.tiff', 'reg_view_111_patch_2_g_6.tiff', 'reg_view_38_patch_2_g_18.tiff', 'reg_view_118_patch_2_g_24.tiff', 'reg_view_71_patch_2_g_29.tiff', 'reg_view_71_patch_2_g_24.tiff', 'reg_view_38_patch_2_g_12.tiff', 'reg_view_161_patch_2_g_9.tiff', 'reg_view_111_patch_2_g_12.tiff', 'reg_view_113_patch_2_g_3.tiff', 'reg_view_113_patch_2_g_16.tiff', 'reg_view_65_patch_2_g_24.tiff', 'reg_view_118_patch_2_g_6.tiff', 'reg_view_115_patch_2_g_2.tiff', 'reg_view_115_patch_2_g_16.tiff', 'reg_view_38_patch_2_g_8.tiff', 'reg_view_38_patch_2_g_2.tiff', 'reg_view_161_patch_2_g_26.tiff', 'reg_view_161_patch_2_g_11.tiff', 'reg_view_113_patch_2_g_20.tiff', 'reg_view_113_patch_2_g_1.tiff', 'reg_view_108_patch_2_g_19.tiff', 'reg_view_65_patch_2_g_0.tiff', 'reg_view_113_patch_2_g_25.tiff', 'reg_view_83_patch_2_g_19.tiff', 'reg_view_38_patch_2_g_0.tiff', 'reg_view_118_patch_2_g_16.tiff', 'reg_view_83_patch_2_g_16.tiff', 'reg_view_155_patch_2_g_22.tiff', 'reg_view_115_patch_2_g_6.tiff', 'reg_view_111_patch_2_g_2.tiff']