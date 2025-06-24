import os
import tifffile as tiff
import shutil
from skimage import io

im_folder = '..//datasets_rush3d_001_02_view_83'+'//'+'001_02_view_83'

for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
    if '.tif' in im_name:
        raw_im_name = im_name
        im_name = im_name.replace('.tif','')
        im_name = im_name.replace('realign_001_02_1x1_80.0ms_Full_Hardware_LaserCount1_240125211112_v_83.','')
        # print(im_name)



        im_dir = im_folder+'//'+raw_im_name
        im = tiff.imread(im_dir)

        im1 = im[:im.shape[0]//2,:,:]
        im2 = im[im.shape[0]//2:,:,:]
        print(im1.shape, im2.shape)

        output_folder1 = im_folder+'_'+im_name+'_1'
        print(output_folder1)

        output_folder2 = im_folder+'_'+im_name+'_2'
        print(output_folder2)

        if not os.path.exists(output_folder1): 
            os.mkdir(output_folder1)
        if not os.path.exists(output_folder2): 
            os.mkdir(output_folder2)

        raw_im_name1 = raw_im_name.replace('.tif','_1.tif')
        raw_im_name2 = raw_im_name.replace('.tif','_2.tif')
        print(raw_im_name1)
        print(raw_im_name2)
        io.imsave(output_folder1+'//'+raw_im_name1, im1)
        io.imsave(output_folder2+'//'+raw_im_name2, im2)

        # if not os.path.exists(output_folder): 
        #     os.mkdir(output_folder)
        
        # shutil.move(im_folder+'//'+raw_im_name, output_folder+'//'+raw_im_name)