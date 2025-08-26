import os
import numpy as np
from skimage import io
import yaml
from thop import profile


#########################################################################
#########################################################################
def get_netpara(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_netparaflops(model, input):
    flops, params = profile(model, inputs=(input,))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


def save_para_dict(save_path, para_dict):
    if '.yaml' in save_path:
        with open(save_path, "w") as yaml_file:
            yaml.dump(para_dict, yaml_file)

    if '.txt' in save_path:
        para_dict_key = para_dict.keys()
        with open(save_path, 'w') as f:  # 设置文件对象
            for now_key in para_dict_key:
                now_para = para_dict[now_key]
                # print(now_key,' -----> ',now_para)
                now_str = now_key + " : " + str(now_para) + '\n'
                f.write(now_str)


def save_img(all_img, norm_factor, output_path, im_name, tag='', if_nor=1):
    all_img = all_img.squeeze().astype(np.float32) * norm_factor
    all_img_name = output_path + '/' + im_name + tag + '.tif'
    if if_nor:
        all_img = all_img - np.min(all_img)
        all_img = all_img / np.max(all_img) * 65535
        
        all_img = np.clip(all_img, 0, 65535).astype('uint16')
    '''
    , if_cut=0
    if if_cut:
        all_img = all_img[0:-opt.img_s+1,:,:]
    '''
    io.imsave(all_img_name, all_img)


def save_img_train(u_in_all, output_path, epoch, index, input_name_list, norm_factor, label_name):
    u_in_path = output_path + '//' + label_name
    if not os.path.exists(u_in_path):
        os.mkdir(u_in_path)
    # print('-----> ',u_in_all.shape)
    for u_in_i in range(0, u_in_all.shape[0]):
        input_name = os.path.basename(input_name_list[u_in_i])
        u_in = u_in_all[u_in_i, :, :, :]
        u_in = u_in.cpu().detach().numpy()
        u_in = u_in.squeeze()

        u_in = u_in.squeeze().astype(np.float32) * norm_factor
        u_in_name = u_in_path + '//' + str(epoch) + '_' + str(index) + '_' + str(
            u_in_i) + '_' + input_name + '_' + label_name + '.tif'
        # print(label_name,' -----> ', u_in.max(), u_in.min())
        io.imsave(u_in_name, u_in)


def UseStyle(string, mode='', fore='', back=''):
    STYLE = {
        'fore':
            {  # 前景色
                'black': 30,  # 黑色
                'red': 31,  # 红色
                'green': 32,  # 绿色
                'yellow': 33,  # 黄色
                'blue': 34,  # 蓝色
                'purple': 35,  # 紫红色
                'cyan': 36,  # 青蓝色
                'white': 37,  # 白色
            },

        'back':
            {  # 背景
                'black': 40,  # 黑色
                'red': 41,  # 红色
                'green': 42,  # 绿色
                'yellow': 43,  # 黄色
                'blue': 44,  # 蓝色
                'purple': 45,  # 紫红色
                'cyan': 46,  # 青蓝色
                'white': 47,  # 白色
            },

        'mode':
            {  # 显示模式
                'mormal': 0,  # 终端默认设置
                'bold': 1,  # 高亮显示
                'underline': 4,  # 使用下划线
                'blink': 5,  # 闪烁
                'invert': 7,  # 反白显示
                'hide': 8,  # 不可见
            },

        'default':
            {
                'end': 0,
            },
    }

    mode = '%s' % STYLE['mode'][mode] if mode in STYLE['mode'] else ''  # .has_key(mode) else ''
    fore = '%s' % STYLE['fore'][fore] if fore in STYLE['fore'] else ''  # .has_key(fore) else ''
    back = '%s' % STYLE['back'][back] if back in STYLE['back'] else ''  # .has_key(back) else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s' % (style, string, end)
