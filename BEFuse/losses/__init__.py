
import torch.nn.functional as F
import torch
from math import exp
from torch import nn
import torch.nn as nn
import torchvision.models as models

from Networks.net import RTFNet_dense

# 确保你已经安装了PyTorch和TorchVision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradient_loss(vi, ir, fu, weights=[10, 10], device=None):
    gradient_Loss = Fusion_loss(vi, ir, fu)
    return gradient_Loss

def perceptual_loss(input,vi,fu):
    perceptual_Loss = PerceptualLoss().cuda()
    percept_Loss = perceptual_Loss(input,vi,fu)
    return percept_Loss

def ssim_loss_vi (fused_result,input_vi ,mask=None):
    ssim_loss=ssim(img1=fused_result,img2=input_vi,mask=mask)

    return ssim_loss

def ssim_loss_ir (fused_result,input_ir ,mask=None):
    ssim_loss_ir=ssim(fused_result,input_ir,mask)

    return ssim_loss_ir


def sf_loss_vi (fused_result,input_vi):
    SF_loss= torch.norm(sf(fused_result)-sf(input_vi))

    return SF_loss

def sf_loss_ir (fused_result,input_ir):
    SF_loss= torch.norm(sf(fused_result)-sf(input_ir))

    return SF_loss

def sf(f1,kernel_radius=5):

    device = torch.device('cuda:0')
    b, c, h, w = f1.shape
    r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
        .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
    return  1-f1_sf

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_uniform_window(window_size, channel=1):
    window = torch.ones((channel, 1, window_size, window_size)) / (window_size * window_size)
    window = window.cuda()
    return window

def ssim(img1, img2, mask=None, window_size=11,window=None, val_range=None):

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if mask is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    else:
        window = create_uniform_window(window_size, channel)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)




    if mask is None:
        ret = ssim_map.mean()
        # print('no_mask_ret',ret)
        return 1-ret
    else:
        # print("mask的数值是",mask.sum())
        ret = (ssim_map * mask).sum() / (mask.sum() + 1e-6)
        # print('mask_ret',ret)
        return 1-ret



def Fusion_loss(vi, ir, fu, weights=[10,2], device=None):

    vi_gray = torch.mean(vi, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device).cuda()

    # 梯度损失
    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    # ## 强度损失
    x_in_max = torch.max(vi, ir)
    loss_in = F.l1_loss(x_in_max, fu)

    # loss_intensity = torch.mean(torch.pow((fu - vi), 2)) + torch.mean((fu_gray < ir) * torch.abs((fu_gray - ir)))
    #
    loss_total = weights[0] * loss_grad + weights[1]*loss_in

    return  loss_total

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=11):
        super(PerceptualLoss, self).__init__()

        RTF = RTFNet_dense(n_class=9)
        pretrained_weight = torch.load(
            "D:/python_project/final_network_2024.12.28/RTF_ckpt/147.pth",
            map_location=lambda storage, loc: storage.cuda(0))
        own_state = RTF.state_dict()
        for name, param in pretrained_weight.items():
            name = name.replace('module.', '')
            if name not in own_state:
                continue
            else:
                own_state[name].copy_(param)


        # vgg = models.vgg19(pretrained=True).features
        # # 只保留到指定层的部分
        # self.vgg = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()

        self.submodel =  RTF.get_submodel(3).eval()
        # 将VGG的参数设置为不进行梯度更新
        for param in self.submodel.parameters():
            param.requires_grad = False

    def forward(self, input,vi,out):
        # 在特征空间中计算L2损失
        out_vi = torch.cat((out,vi),dim=1)
        out_vi = self.submodel(out_vi)
        ir_vi = self.submodel(input)

        loss = nn.functional.mse_loss(out_vi,ir_vi)
        return loss

if __name__ == '__main__':
    ppp = PerceptualLoss()
    input = torch.ones(1,1,480,640)
    out = ppp(input,input)
    print(out)






