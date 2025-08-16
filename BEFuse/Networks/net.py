import gc
import os

import numpy as np
import torch
import math

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from torch.utils.data import DataLoader
from torchsummary import summary
from pytorch_grad_cam import GradCAM
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision import transforms
import torchvision.models as models
# from pytorch_grad_cam.utils.ircam import func_ircam
import torch.optim as optim


from torchinfo import summary
from skimage import exposure


#=================================RTFNet=======================================================================


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # print('(model_output[self.category, :, :] * self.mask).sum()的值是')
        return (model_output[self.category, :, :] * self.mask).sum()

class RTFNet_dense(nn.Module):
    def __init__(self, n_class):
        super(RTFNet_dense, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 152:
            resnet_raw_model1 = models.densenet121(pretrained=True)
            self.inplanes = 1024

        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 对ResNet模型的conv0层权重在通道维度（dim=1）求平均，得到单通道
        average_weight = torch.mean(resnet_raw_model1.features.conv0.weight.data, dim=1, keepdim=True)

        # 使用 repeat 在通道维度上复制，扩展为两个通道
        # (out_channels, 1, kernel_height, kernel_width) -> (out_channels, 2, kernel_height, kernel_width)
        repeated_weight = average_weight.repeat(1, 2, 1, 1)

        # 将扩展后的权重赋值给encoder_thermal_conv1
        self.encoder_thermal_conv1.weight.data = repeated_weight

        self.encoder_thermal_bn1 = resnet_raw_model1.features.norm0
        self.encoder_thermal_relu = resnet_raw_model1.features.relu0
        self.encoder_thermal_maxpool = resnet_raw_model1.features.pool0

        self.encoder_thermal_layer1 = resnet_raw_model1.features._modules.get('denseblock1')
        self.encoder_thermal_trans1 = resnet_raw_model1.features._modules.get('transition1')
        self.encoder_thermal_layer2 = resnet_raw_model1.features._modules.get('denseblock2')
        self.encoder_thermal_trans2 = resnet_raw_model1.features._modules.get('transition2')
        self.encoder_thermal_layer3 = resnet_raw_model1.features._modules.get('denseblock3')
        self.encoder_thermal_trans3 = resnet_raw_model1.features._modules.get('transition3')
        self.encoder_thermal_layer4 = resnet_raw_model1.features._modules.get('denseblock4')

        ########  DECODER  ########

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)

    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )

        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):

        thermal = input
        # print(thermal.shape)
        # thermal = input
        verbose = False

        # encoder
        ######################################################################
        # if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose: print("thermal.size() original: ", thermal.size())  # (480, 640)
        ####初始化层##################################################################
        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size(), thermal.shape)  # (120, 160)
        #####第一层#################################################################
        thermal = self.encoder_thermal_layer1(thermal)
        thermal = self.encoder_thermal_trans1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size())  # (120, 160)
        ######第二层################################################################
        thermal = self.encoder_thermal_layer2(thermal)
        thermal = self.encoder_thermal_trans2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)
        #######第三层###############################################################
        thermal = self.encoder_thermal_layer3(thermal)
        thermal = self.encoder_thermal_trans3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)
        #######第四层##############################################################
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)
        fuse = thermal
        ######################################################################

        # decoder
        fuse = self.deconv1(fuse)
        if verbose: print("fuse after deconv1: ", fuse.size())  # (30, 40)
        fuse = self.deconv2(fuse)
        if verbose: print("fuse after deconv2: ", fuse.size())  # (60, 80)
        fuse = self.deconv3(fuse)
        if verbose: print("fuse after deconv3: ", fuse.size())  # (120, 160)
        fuse = self.deconv4(fuse)
        if verbose: print("fuse after deconv4: ", fuse.size())  # (240, 320)
        fuse = self.deconv5(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size())  # (480, 640)

        return fuse
    def get_submodel(self, feature_layer):
        layers = []

        # 添加编码器层
        layers.append(self.encoder_thermal_conv1)
        layers.append(self.encoder_thermal_bn1)
        layers.append(self.encoder_thermal_relu)
        layers.append(self.encoder_thermal_maxpool)

        # 根据 feature_layer 的值，添加相应的层
        if feature_layer >= 1:
            layers.append(self.encoder_thermal_layer1)
        if feature_layer >= 2:
            layers.append(self.encoder_thermal_trans1)
        if feature_layer >= 3:
            layers.append(self.encoder_thermal_layer2)
        if feature_layer >= 4:
            layers.append(self.encoder_thermal_trans2)
        if feature_layer >= 5:
            layers.append(self.encoder_thermal_layer3)
        if feature_layer >= 6:
            layers.append(self.encoder_thermal_trans3)
        if feature_layer >= 7:
            layers.append(self.encoder_thermal_layer4)
        # 构建子模型
        submodel = nn.Sequential(*layers).eval()
        return submodel

class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out






#=================================two—Stream============================================================================
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)

        return x,x1+x2

class Cross_encoder(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super(Cross_encoder,self).__init__()
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=1, with_bn=True)
        self.act = nn.ReLU6()
    def forward(self, x1,x2):
        x = self.act(x1) * self.act(x2)
        x = self.dwconv2(self.g(x))
        x = self.act(x)

        return x

class Encoder(nn.Module):
    def __init__(self,ch,depths=[1]):
        super(Encoder, self).__init__()
        self.vis_conv = ConvLeakyRelu2d(1, ch[1])
        self.block = nn.ModuleList([Block(ch[1]) for _ in range(depths[0])])

    def forward(self,x):
        x1=self.vis_conv(x)
        for blk in self.block:
            x1,x_sum= blk(x1)
        return x1,x_sum

class Decoder(nn.Module):
    def __init__(self,vis_ch,inf_ch,output):
        super(Decoder, self).__init__()
        self.decode3 = ConvBnLeakyRelu2d(32, 16)
        self.decode2 = ConvBnLeakyRelu2d(16, 8)
        self.decode1 = ConvBnTanh2d(8, output)
    def forward(self,x):
        x1=self.decode3(x)
        x2=self.decode2(x1)
        x3=self.decode1(x2)
        return x3
#################################################################################################################################################################
class MODEL(nn.Module):
    def __init__(self, vi_in_channel=2, num_init_features=32, growth_rate=16, block_config=(2, 4, 5), se_ratio=8,
                 is_train=True):
        super(MODEL, self).__init__()
        self.is_train = is_train


        ########################################################################################################################
        self.RTF = RTFNet_dense(n_class=9)
        pretrained_weight = torch.load(
            "D:/python_project/final_network_2024.12.28/RTF_ckpt/147.pth",
            map_location=lambda storage, loc: storage.cuda(0))
        own_state = self.RTF.state_dict()
        for name, param in pretrained_weight.items():
            name = name.replace('module.', '')
            if name not in own_state:
                print("没有导入",name)
                continue
            else:
                own_state[name].copy_(param)
                print("成功导入",name)
        print('RTF done!')
        ########################################################################################################################

        vis_ch = [16, 32, 48]
        inf_ch = [16, 32, 48]
        output = 1
    #============================================================================
        self.cross_encoder = Cross_encoder(vis_ch[1])
    #=============================================================================
        self.vi_encoder = Encoder(vis_ch)
        self.ir_encoder = Encoder(inf_ch)

        self.decoder = Decoder(vis_ch,inf_ch,output)

        if self.is_train:
            self.RTF = nn.DataParallel(self.RTF)
        self.cross_encoder = nn.DataParallel(self.cross_encoder)
        self.vi_encoder = nn.DataParallel(self.vi_encoder)
        self.ir_encoder = nn.DataParallel(self.ir_encoder)
        self.decoder = nn.DataParallel(self.decoder)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input):
        vi = input[:, :1]
        ir = input[:, 1:]

        # 将self.RTF和ir传进一个可以显示cam的方法,
        if self.is_train:
            ir_cam_0 = self.func_ircam(self.RTF, input)

            ir_cam_1 = torch.where(ir_cam_0 < 0.34, torch.tensor(0.0, device=ir_cam_0.device), ir_cam_0)
            ir_cam_2 = torch.where(ir_cam_1 >= 0.34, torch.tensor(1.0, device=ir_cam_1.device), ir_cam_1)

        vi_enc_output,x1_sum = self.vi_encoder(vi)
        ir_enc_output,x2_sum = self.ir_encoder(ir)
        cross_input = self.cross_encoder(x1_sum,x2_sum)
        input_decoder = vi_enc_output*ir_enc_output + cross_input
        out = self.decoder(input_decoder)

        if self.is_train:
            return out, ir_cam_2
        else:
            return out

    def visualize_activations(self, input_tensor, pit_name, save_path="activations"):
        """
        可视化并保存关键层的激活
        - 对于vi_encoder和ir_encoder：保存所有特征图通道
        - 对于其他层：保存通道平均值
        无坐标轴和colorbar的简洁版本
        """
        save_path = os.path.join(save_path, pit_name)
        os.makedirs(save_path, exist_ok=True)

        # 存储各层输出
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                # 处理多输出情况
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()  # 取元组的第一个元素
                else:
                    activations[name] = output.detach()

            return hook

        # 注册hook
        hooks = []
        layers_to_visualize = {
            'vi_encoder': self.vi_encoder.module,
            'ir_encoder': self.ir_encoder.module,
            'cross_encoder': self.cross_encoder.module.act,
            'decoder': self.decoder.module.decode1.conv
        }

        for name, layer in layers_to_visualize.items():
            hooks.append(layer.register_forward_hook(get_activation(name)))

        # 前向传播
        with torch.no_grad():
            self(input_tensor)

        # 移除hook
        for hook in hooks:
            hook.remove()

        # 可视化并保存
        for name, act in activations.items():
            act = act.cpu().numpy()  # [batch, channels, H, W]

            if name in ["vi_encoder","cross_encoder"]:
                # 为编码器创建单独文件夹
                encoder_path = os.path.join(save_path, name)
                os.makedirs(encoder_path, exist_ok=True)

                # 获取第一个样本的所有通道
                sample_act = act[0]  # [channels, H, W]
                num_channels = sample_act.shape[0]

                if name == "vi_encoder":
                    # 创建专门文件夹
                    channel_dir = os.path.join(encoder_path, "vi_selected_channels")
                    os.makedirs(channel_dir, exist_ok=True)

                    # 获取目标通道数据
                    channel_1 = sample_act[16]
                    channel_13 = sample_act[2]
                    channel_8 = sample_act[19]
                    channel_20= sample_act[29]


                    plt.figure(figsize=(12, 10))
                    plt.imshow(channel_1,
                               cmap='inferno',
                               vmin=np.percentile(channel_1, 5),
                               vmax=np.percentile(channel_1, 95),
                               interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_dir, 'channel_1.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

                    plt.figure(figsize=(12, 10))
                    plt.imshow(channel_20,
                               cmap='inferno',
                               vmin=np.percentile(channel_20, 5),
                               vmax=np.percentile(channel_20, 95),
                               interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_dir, 'channel_20.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

                    # ===== 1. 单独保存通道13 =====
                    plt.figure(figsize=(12, 10))
                    plt.imshow(channel_13,
                               cmap='inferno',
                               vmin=np.percentile(channel_13, 5),
                               vmax=np.percentile(channel_13, 95),
                               interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_dir, 'channel_13.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

                    # ===== 2. 单独保存通道20 =====
                    plt.figure(figsize=(12, 10))
                    plt.imshow(channel_8,
                               cmap='inferno',
                               vmin=np.percentile(channel_8, 5),
                               vmax=np.percentile(channel_8, 95),
                               interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_dir, 'channel_8.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

                    # ===== 3. 保存合并结果 =====
                    plt.figure(figsize=(12, 10))
                    merged = 0.3 * channel_13 +  0.3 * channel_1+ 0.4*channel_20+0.1*channel_8# 加权融合
                    plt.imshow(merged,
                               cmap='inferno',
                               vmin=np.percentile(merged, 5),
                               vmax=np.percentile(merged, 95),
                               interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_dir, 'merged_13_20.png'),
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()



                else:# 保存每个通道的特征图
                    for ch_idx in range(num_channels):
                        plt.figure(figsize=(10, 8))

                        plt.imshow(sample_act[ch_idx],
                                   cmap='inferno',
                                   vmin=np.percentile(sample_act[ch_idx], 5),
                                   vmax=np.percentile(sample_act[ch_idx], 95),
                                   interpolation='nearest')
                        plt.axis('off')

                        # 保存单个通道
                        save_file = os.path.join(encoder_path, f'channel_{ch_idx}.png')
                        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close()

                    print(f"Saved {num_channels} channel visualizations to: {encoder_path}")

                    # 可选：保存所有通道的网格图
                    plt.figure(figsize=(20, 20))
                    cols = int(np.ceil(np.sqrt(num_channels)))
                    for ch_idx in range(num_channels):
                        plt.subplot(cols, cols, ch_idx + 1)

                        plt.imshow(sample_act[ch_idx],
                                   cmap='inferno',
                                   vmin=np.percentile(sample_act[ch_idx], 5),
                                   vmax=np.percentile(sample_act[ch_idx], 95),
                                   interpolation='nearest')
                        plt.axis('off')
                        plt.title(f'Ch{ch_idx}', fontsize=8)
                    plt.savefig(os.path.join(encoder_path, 'all_channels.png'),
                                bbox_inches='tight', dpi=150)
                    plt.close()


            else:
                # 对其他层仍使用平均值可视化
                mean_act = act.mean(axis=1)[0]  # 计算所有通道的平均 [H, W]

                plt.figure(figsize=(10, 8))
                plt.imshow(mean_act,
                           cmap='inferno',
                           vmin=np.percentile(mean_act, 5),
                           vmax=np.percentile(mean_act, 95),
                           interpolation='nearest')
                plt.axis('off')

                save_file = os.path.join(save_path, f'{name}_mean_activation.png')
                plt.savefig(save_file, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()

                print(f"Saved mean activation visualization to: {save_file}")


    def func_ircam(self, RTF, ir):
        RTF.eval()

        output = RTF(ir)

        normalized_masks = torch.nn.functional.softmax(output, dim=1)

        sem_classes = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]

        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

        # category = sem_class_to_idx["person"]

        mask = normalized_masks[:, :, :, :].argmax(axis=1).detach()

        car_mask_float = (mask).float()

        target_layers = [RTF.module.deconv5]

        category = [1,2,4]

        targets = [SemanticSegmentationTarget(category, car_mask_float[i, :, :]) for i in range(ir.shape[0])]
        # print(len(targets))
        with GradCAM(model=RTF,
                     target_layers=target_layers,
                     ) as cam:
            grayscale_cam = cam(input_tensor=ir,
                                targets=targets)
        # print('grayscale_cam.sum()的数值为', grayscale_cam.sum())
        return grayscale_cam



if __name__ == '__main__':

    input = torch.ones(2,2,480,640).cuda()
    model = MODEL(is_train=False)
    model.cuda()
    summary(model, input_size=(1,2,480,640))
    model.visualize_activations(input)
    # output = model(input)
    # print(output.shape)
