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


def img_save(img_array, name, output_path):
    # 确保图像形状为 (H, W, 3)
    if img_array.ndim == 3 and img_array.shape[0] == 3:
        img_array = img_array.transpose(1, 2, 0)
    img = Image.fromarray(img_array.astype(np.uint8))
    os.makedirs(output_path, exist_ok=True)
    img.save(os.path.join(output_path, f'{name}.png'))


def fusion(image_names, path1='./source images/', output_path='./fusion result/'):
    tran = transforms.ToTensor()

    for name in image_names:
        # 处理可见光图像
        if 'Kaptein' in name[0]:
            file_vi_path = os.path.join(path1, f'{name[0]}.bmp')
            img_vi = Image.open(file_vi_path)
            vi_Y = tran(img_vi).unsqueeze(0).to(device)
        elif 'meting'in name[0]:
            file_vi_path = os.path.join(path1, f'{name[0]}.bmp')
            img_vi = Image.open(file_vi_path)
            vi_Y = tran(img_vi).unsqueeze(0).to(device)
        elif 'FLIR' in name[0]:
            file_vi_path = os.path.join(path1, f'{name[0]}.jpg')
            img_vi = Image.open(file_vi_path).convert('RGB')
            img_vi = tran(img_vi).unsqueeze(0).to(device)
            vi_Y, vi_Cr, vi_Cb = RGB2YCrCb(img_vi)
        elif 'TNO' in name[0]:
            file_vi_path = os.path.join(path1, f'{name[0]}.png')
            img_vi = Image.open(file_vi_path).convert('L')
            vi_Y = tran(img_vi).unsqueeze(0).to(device)
        else:
            file_vi_path = os.path.join(path1, f'{name[0]}.png')
            img_vi = Image.open(file_vi_path).convert('RGB')
            img_vi = tran(img_vi).unsqueeze(0).to(device)
            print(name[0],'的通道数为',img_vi.shape)
            vi_Y, vi_Cr, vi_Cb = RGB2YCrCb(img_vi)

        # 处理红外图像
        if 'Kaptein' in name[1]:
            file_ir_path = os.path.join(path1, f'{name[1]}.bmp')
        elif 'meting' in name[1]:
            file_ir_path = os.path.join(path1, f'{name[1]}.bmp')
        elif 'FLIR' in name[1]:
            file_ir_path = os.path.join(path1, f'{name[1]}.jpg')
        else:
            file_ir_path = os.path.join(path1, f'{name[1]}.png')
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
            model.visualize_activations(input,name[0])
            # visualize_feature_maps(model, input)

        print(f"Model output shape: {out.shape}")

        if 'meting' in name[0]:

            fused_img = out.squeeze(0)
            print(f"Fused image shape (after squeeze): {fused_img.shape}")

            out_img = np.transpose((fused_img * 255).detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            print(f"Output image shape for saving: {out_img.shape}")
            out_img = out_img.squeeze(2)
            img_save(out_img, f'output_fuse_image_vi64_{name[0]}', output_path)

        elif 'Kaptein' in name[0] or 'TNO' in name[0]:
            fused_img = out.squeeze(0)
            print(f"Fused image shape (after squeeze): {fused_img.shape}")

            out_img = np.transpose((fused_img * 255).detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            print(f"Output image shape for saving: {out_img.shape}")
            out_img = out_img.squeeze(2)
            img_save(out_img, f'output_fuse_image_vi64_{name[0]}', output_path)

        else:
            # 将融合结果与CrCb还原成RGB
            fused_img = YCbCr2RGB(out, vi_Cr, vi_Cb)

            print(f"Fused image shape (before squeeze): {fused_img.shape}")
            fused_img = fused_img.squeeze(0)
            print(f"Fused image shape (after squeeze): {fused_img.shape}")
            print('hahahhahaha')
            out_img = np.transpose((fused_img * 255).detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            print(f"Output image shape for saving: {out_img.shape}")

            img_save(out_img, f'output_fuse_image_vi64_{name[0]}', output_path)

if __name__ == '__main__':

    model = net(is_train=False)
    model.cuda()
    model_path = "./models/model_name/model_120.pth" #可以关注一下model_name257/model_50.pth"，因为它的纹理细节过多  还有loss ssim_target_loss  ssim_all_loss 4gradient的model_9.pth和model_20.pth和model_21.pth
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
    print("11的轮参数效果最好")


    tic = time.time()
    image_list=[("e00706N_vi","e00706N_ir")]
    # image_list = [("01205N_rgb","01205N_th"),("00521D_rgb","00521D_th"),("01196N_rgb","01196N_th"),("00559D_rgb","00559D_th"),("01042N_rgb","01042N_th"),("00001D_rgb","00001D_th"),("01324N_rgb","01324N_th"),("00195D_rgb","00195D_th"),("00153D_rgb","00153D_th"),("00140D_rgb","00140D_th"),("00028N_rgb","00028N_th"),("00633D_rgb","00633D_th"),("01023N_rgb","01023N_th"),("00881N_rgb","00881N_th"),("00858N_rgb","00858N_th"),("00585D_rgb","00585D_th"),("00537D_rgb","00537D_th")]
    for i in range(len(image_list)):
        image_pairs = [image_list[i]]
        fusion(image_pairs,path1='./MSRS/')
    toc = time.time()
    print('time: {}'.format(toc - tic))



    # tic = time.time()
    # TNO_images = [('TNOvi_01', 'TNOir_01'), ('TNOvi_02', 'TNOir_02'), ('TNOvi_03', 'TNOir_03'),
    # ('TNOvi_04', 'TNOir_04'), ('TNOvi_05', 'TNOir_05'), ('TNOvi_06', 'TNOir_06'),
    # ('TNOvi_07', 'TNOir_07'), ('TNOvi_08', 'TNOir_08'), ('TNOvi_09', 'TNOir_09'),
    # ('TNOvi_10', 'TNOir_10'), ('TNOvi_11', 'TNOir_11'), ('TNOvi_12', 'TNOir_12'),
    # ('TNOvi_13', 'TNOir_13'), ('TNOvi_14', 'TNOir_14'), ('TNOvi_15', 'TNOir_15'),
    # ('TNOvi_16', 'TNOir_16'), ('TNOvi_17', 'TNOir_17'), ('TNOvi_18', 'TNOir_18'),
    # ('TNOvi_19', 'TNOir_19'), ('TNOvi_20', 'TNOir_20'), ('TNOvi_21', 'TNOir_21'),
    # ('TNOvi_22', 'TNOir_22'), ('TNOvi_23', 'TNOir_23'), ('TNOvi_24', 'TNOir_24'),
    # ('TNOvi_25', 'TNOir_25'), ('TNOvi_26', 'TNOir_26'), ('TNOvi_27', 'TNOir_27'),
    # ('TNOvi_28', 'TNOir_28'), ('TNOvi_29', 'TNOir_29'), ('TNOvi_30', 'TNOir_30'),
    # ('TNOvi_31', 'TNOir_31'), ('TNOvi_32', 'TNOir_32'), ('TNOvi_33', 'TNOir_33'),
    # ('TNOvi_34', 'TNOir_34'), ('TNOvi_35', 'TNOir_35'), ('TNOvi_36', 'TNOir_36'),
    # ('TNOvi_37', 'TNOir_37'), ('TNOvi_38', 'TNOir_38'), ('TNOvi_39', 'TNOir_39'),
    # ('TNOvi_40', 'TNOir_40'), ('TNOvi_41', 'TNOir_41'), ('TNOvi_42', 'TNOir_42')]
    # for i in range(len(TNO_images)):
    #     image_pairs = [TNO_images[i]]
    #     fusion(image_pairs)
    # toc = time.time()
    # print('time: {}'.format(toc - tic))