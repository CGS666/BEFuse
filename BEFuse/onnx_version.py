import torch
from onnx_model.onnx_net import MODEL
from torch import nn
import torch.nn.functional as F
import onnx
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from collections import OrderedDict
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from tqdm import tqdm
model = MODEL(is_train=False)
model.cuda()
model_path = "D:/python_project/final_network_2024.12.28/models/model_name/model_120.pth" #可以关注一下model_name257/model_50.pth"，因为它的纹理细节过多  还有loss ssim_target_loss  ssim_all_loss 4gradient的model_9.pth和model_20.pth和model_21.pth
use_gpu = torch.cuda.is_available()
state_dict = torch.load(model_path)

# 初始化新的 state_dict
new_state_dict = OrderedDict()

own_state = model.state_dict()

# 判断是否使用 GPU
if use_gpu:
    for name, param in state_dict.items():
        name = name.replace('module.', '')
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
model = model.eval()

print('ONNX 版本', onnx.__version__)
x = torch.randn(1, 2, 768, 1024).cuda()
output = model(x)
output.shape

with torch.no_grad():
    torch.onnx.export(
        model,                       # 要转换的模型
        x,                           # 模型的任意一组输入
        'BEFuse.onnx',    # 导出的 ONNX 文件名
        opset_version=11,            # ONNX 算子集版本
        input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
        output_names=['output']      # 输出 Tensor 的名称（自己起名字）
    )

onnx_model = onnx.load('BEFuse.onnx')
onnx.checker.check_model(onnx_model)


def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    # Clamp the values to the range [0, 1] for Y, Cr, and Cb
    Y = np.clip(Y, 0.0, 1.0)
    Cr = np.clip(Cr, 0.0, 1.0)
    Cb = np.clip(Cb, 0.0, 1.0)

    return Y, Cr, Cb


def YCbCr2RGB(Y, Cr, Cb):
    R = Y + 1.403 * (Cr - 0.5)
    G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
    B = Y + 1.772 * (Cb - 0.5)

    # Combine R, G, and B channels into a single image and clamp to [0, 1]
    RGB = np.concatenate((R, G, B), axis=1)
    return np.clip(RGB, 0.0, 1.0)


def img_save(img_array, name, output_path):
    # 确保图像形状为 (H, W, 3)
    if img_array.ndim == 3 and img_array.shape[0] == 3:
        img_array = img_array.transpose(1, 2, 0)
    img = Image.fromarray(img_array.astype(np.uint8))
    os.makedirs(output_path, exist_ok=True)
    img.save(os.path.join(output_path, f'{name}.png'))


def vi_test_transform(vi_rgb):
    tran = transforms.ToTensor()
    img_vi = vi_rgb.convert('RGB')
    img_vi = tran(img_vi).unsqueeze(0).numpy()
    vi_Y, vi_Cr, vi_Cb = RGB2YCrCb(img_vi)
    return vi_Y, vi_Cr, vi_Cb


def ir_test_transform(ir_rgb):
    tran = transforms.ToTensor()
    img_ir = ir_rgb.convert('L')
    img_ir = tran(img_ir).unsqueeze(0).numpy()
    return img_ir


def process_frame(vi_bgr, ir_brg,o):
    # 记录该帧开始处理的时间
    start_time = time.time()
    vi_rgb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2RGB)  # BGR转RGB
    #     ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB) # BGR转RGB

    vi_img_pil = Image.fromarray(vi_rgb)  # array 转 PIL
    ir_img_pil = Image.fromarray(ir_brg)  # array 转 PIL

    ## 预处理
    vi_Y, vi_Cr, vi_Cb = vi_test_transform(vi_img_pil)  # 预处理
    print(vi_Y.shape)
    img_ir = ir_test_transform(ir_img_pil)
    print(img_ir.shape)
    input = np.concatenate((vi_Y, img_ir), axis=1)
    print(input.shape)

    ## onnx runtime 预测
    ort_inputs = {'input': input}  # onnx runtime 输入
    out = ort_session.run(['output'], ort_inputs)[0]  # onnx runtime 输出
    print(out)
    fused_img = YCbCr2RGB(out, vi_Cr, vi_Cb)
    fused_img = fused_img.squeeze(0)
    out_img = np.transpose((fused_img * 255), (1, 2, 0)).astype(np.uint8)

    img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)  # RGB转BGR

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    return img_bgr


def generate_video(vi_path='video/vi_vedio.mp4', ir_path='video/ir_vedio.mp4'):
    filehead = vi_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', vi_path)
    # 获取视频总帧数
    vi_cap = cv2.VideoCapture(vi_path)
    vi_frame_count = 0
    while (vi_cap.isOpened()):
        vi_success, vi_frame = vi_cap.read()
        vi_frame_count += 1
        if not vi_success:
            break
    vi_cap.release()
    print('vi视频总帧数为', vi_frame_count)

    print('视频开始处理', ir_path)
    # 获取视频总帧数
    ir_cap = cv2.VideoCapture(ir_path)
    ir_frame_count = 0
    while (ir_cap.isOpened()):
        ir_success, ir_frame = ir_cap.read()
        ir_frame_count += 1
        if not ir_success:
            break
    ir_cap.release()
    print('ir视频总帧数为', ir_frame_count)

    vi_cap = cv2.VideoCapture(vi_path)
    ir_cap = cv2.VideoCapture(ir_path)

    vi_frame_size = (vi_cap.get(cv2.CAP_PROP_FRAME_WIDTH), vi_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = vi_cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(vi_frame_size[0]), int(vi_frame_size[1])))
    # 进度条绑定视频总帧数
    with tqdm(total=vi_frame_count - 1) as pbar:
        try:
            while (vi_cap.isOpened()):
                vi_success, vi_frame = vi_cap.read()
                ir_success, ir_frame = ir_cap.read()
                if not vi_success:
                    break
                try:
                    frame = process_frame(vi_frame, ir_frame)

                except:
                    print('报错！', error)
                    pass

                if vi_success == True:
                    # cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    vi_cap.release()
    ir_cap.release()
    print('视频已保存', output_path)


generate_video(vi_path='vi_vedio.mp4', ir_path='ir_vedio.mp4')