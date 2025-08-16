from collections import OrderedDict
from skimage.io import imsave
from PIL import Image
import numpy as np
import os
import torch
from torch import nn
import time
import imageio
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from Networks.net import MODEL as net
from collections import OrderedDict


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


def Img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)


def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()

    return Y, Cr, Cb


def YCbCr2RGB(Y, Cr, Cb):
    R = Y + 1.403 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)

    RGB = torch.cat((R, G, B), dim=1)
    return RGB.clamp(0.0, 1.0)

def pair_images(folder1, folder2):
        # 获取文件夹中所有的文件名（不包括路径）
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

        # 去掉文件扩展名
    names1 = [os.path.splitext(file)[0] for file in files1]
    names2 = [os.path.splitext(file)[0] for file in files2]

        # 创建一个列表用于存储配对的结果
    image_list = []

        # 以较小的文件数量为基准进行配对
    min_length = min(len(names1), len(names2))
    for i in range(min_length):
        image_list.append((names1[i], names2[i]))

    return image_list

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)

def img_save(img_array, name, output_path):
    # 确保图像形状为 (H, W, 3)
    if img_array.ndim == 3 and img_array.shape[0] == 3:
        img_array = img_array.transpose(1, 2, 0)
    img = Image.fromarray(img_array.astype(np.uint8))
    os.makedirs(output_path, exist_ok=True)
    img.save(os.path.join(output_path, f'{name}.png'))


def fusion(image_names, path1='D:/python_project/all_dataset/RoadScene-master/crop_LR_visible',path2='D:/python_project/all_dataset/RoadScene-master/cropinfrared', output_path='./RoadScene fusion result/'):
    tran = transforms.ToTensor()

    for name in image_names:

        file_vi_path = os.path.join(path1, f'{name[0]}.png')
        img_vi = Image.open(file_vi_path).convert('RGB')
        img_vi = tran(img_vi).unsqueeze(0).to(device)
        print(name[0],'的通道数为',img_vi.shape)
        vi_Y, vi_Cr, vi_Cb = RGB2YCrCb(img_vi)

        file_ir_path = os.path.join(path2, f'{name[1]}.png')
        img_ir = Image.open(file_ir_path).convert('L')
        img_ir = tran(img_ir).unsqueeze(0).to(device)
        print(name[1], '的通道数为', img_ir.shape)

        # 拼接输入
        input = torch.cat((vi_Y,img_ir), dim=1)

        if use_gpu:
            input = input.cuda()

        model.eval()
        with torch.no_grad():
            out = model(input)
            out = clamp(out)


        fused_img = YCbCr2RGB(out, vi_Cr, vi_Cb)

        print(f"Fused image shape (before squeeze): {fused_img.shape}")
        fused_img = fused_img.squeeze(0)
        print(f"Fused image shape (after squeeze): {fused_img.shape}")
        out_img = np.transpose((fused_img * 255).detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        print(f"Output image shape for saving: {out_img.shape}")

        img_save(out_img, f'{name[0]}', output_path)

if __name__ == '__main__':

    model = net(is_train=False)
    # model.cuda()
    model_path = "./models/model_name/model_120.pth"
    use_gpu = torch.cuda.is_available()
    state_dict = torch.load(model_path)

    # 初始化新的 state_dict
    new_state_dict = OrderedDict()

    own_state = model.state_dict()

    # 判断是否使用 GPU
    if use_gpu:
        for name, param in state_dict.items():
            if name not in own_state or 'RTF' in name:
                print(f"Warning: {name} not found in model state_dict.")
                continue
            else:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
                # print(name)
                print(f"Updated: {name}")
        print('model done!')

    # 使用函数M
    folder1_path = r'D:\python_project\all_dataset\MSRS-main\test\ir'
    folder2_path = r'D:\python_project\all_dataset\MSRS-main\test\vi'
    out_path = './MSRS_test_result'
    image_list = pair_images(folder1_path, folder2_path)
    print(image_list)

    tic = time.time()
    for i in range(len(image_list)):
        image_pairs = [image_list[i]]
        fusion(image_pairs,folder1_path,folder2_path,out_path)
    toc = time.time()
    print('time: {}'.format(toc - tic))


