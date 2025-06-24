import numpy as np

def adjust_time_resolution(input_path,
                            input_folder,
                            output_path,
                            output_folder,
                            t_resolution):
    data_path = input_path+'//'+input_folder
    import os
    for im_name in list(os.walk(data_path, topdown=False))[-1][-1]:
        if '.tif' in im_name:
            import tifffile as tiff
            im_dir = data_path+'//'+im_name
            im = tiff.imread(im_dir)

            im_w = im.shape[2]
            im_h = im.shape[1]
            im_s = im.shape[0]

            new_im_s = int(im_s//(t_resolution/10))
            new_im = np.zeros((new_im_s, im_h, im_w))
            # print(new_im.shape)

            im_s_list = np.linspace(0, im_s, im_s)
            new_im_s_list = np.linspace(0, im_s, new_im_s)
            for i in range(0, im_h):
                for ii in range(0, im_w):
                    new_im[:,i,ii] = np.interp(new_im_s_list, im_s_list, im[:,i,ii])
            
            # print(new_im.shape)
            if not os.path.exists(output_path): 
                os.mkdir(output_path)
            if not os.path.exists(output_path+'//'+output_folder): 
                os.mkdir(output_path+'//'+output_folder)
            from skimage import io
            io.imsave(output_path+'//'+output_folder+'//'+im_name, new_im)



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