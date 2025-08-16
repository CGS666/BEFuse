import os
import argparse

from torch import nn
from torch.nn import MSELoss

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

import glob

from collections import OrderedDict
import torch
import joblib
import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.net import MODEL as net
from losses import ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi,gradient_loss,perceptual_loss
from util.dataset_create import MF_dataset
from util.augmentation import RandomFlip, RandomCrop

#定义训练的设备

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


devices = [try_gpu(i) for i in range(2)]

use_gpu = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model_name', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--weight', default=[1, 0.05, 0.0006, 0.00025], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--data_dir', '-dr', type=str, default='D:/python_project/RTFNet-1/RTFNet-master/dataset/')
    parser.add_argument('--num_workers', '-j', type=int, default=8)

    args = parser.parse_args()

    return args


augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def train(args, train_dataloader, model,loss_func ,optimizer):
def train(args, train_dataloader, model, criterion_ssim_ir_target, criterion_gradient, criterion_perceptual,criterion_ssim_all, optimizer, writer, epoch):
    losses = AverageMeter()
    losses_ssim_ir_target = AverageMeter()
    losses_gradient = AverageMeter()
    losses_ssim_all = AverageMeter()

    model.train()

    for step, input in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        optimizer.zero_grad()


        input = input.cuda()

        vi = input[:, :1]
        ir = input[:, 1:]
        out, ir_cam = model(input)  # 一个batch输入model并更新参数

        # print(vi.device,ir.device,out.device)
        masked_out = out * ir_cam
        masked_original = ir * ir_cam

        back_mask = 1 - ir_cam


        ssim_target_loss = criterion_ssim_ir_target(out, ir*ir_cam,mask=ir_cam) #(out*ir_cam, ir*ir_cam)只有rgb图像  (out, ir*ir_cam,mask=ir_cam)融合成功但是有虚影
                                                                                    #(out, ir)和10*(out,vi)只有RGB图像
        ssim_all_loss = criterion_ssim_all(out, vi)

        gradient = criterion_gradient(vi,ir,out,device=0)

        perceptual = criterion_perceptual(input,vi,out)

        #ssim_target_loss使得红外目标更清楚，ssim_all_loss可见光图像纹理更加美观， gradient使得图像生成来的红外纹理更加清醒没有糊影，perceptual保留两者的语义信息(网络结构会不会太简单没办法拟合这些复杂的信息)

        loss = ssim_target_loss + ssim_all_loss + gradient + perceptual #要在ssim_target和ssim_all_loss中加入自适应参数，用边缘检测

        losses.update(loss.item(), ir.size(0))
        losses_ssim_ir_target.update(ssim_target_loss.item(), ir.size(0))
        losses_ssim_all.update(ssim_all_loss.item(), ir.size(0))

        loss.backward()
        optimizer.step()

        if step%100==0:
        # 在每个步骤记录损失到 TensorBoard
            writer.add_scalar("train/loss", loss, epoch * len(train_dataloader) + step)
            writer.add_scalar("train/loss_gradient", gradient, epoch * len(train_dataloader) + step)
            writer.add_scalar("train/loss_ssim_all", ssim_all_loss, epoch * len(train_dataloader) + step)
            writer.add_scalar("train/loss_perceptual", perceptual, epoch * len(train_dataloader) + step) #可以再加一个target
            writer.add_scalar("train/loss_target", ssim_target_loss, epoch * len(train_dataloader) + step)

    log = OrderedDict([
        # ('Transformer_A1', Transformer_A1),
        # ('Transformer_B1', Transformer_B1),
        ('loss', losses.avg),
        ('loss_ssim_ir_target', losses_ssim_ir_target.avg),
        ('loss_ssim_all', losses_ssim_all.avg),
    ])
    return log


def main():
    args = parse_args()

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    cudnn.benchmark = True
    writer = SummaryWriter("Y-shape_logs")

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = net()

    model.cuda()

    criterion_ssim_all = ssim_loss_ir
    criterion_ssim_ir = ssim_loss_ir
    criterion_gradient = gradient_loss
    criterion_perceptual = perceptual_loss

    optimizer = optim.Adam([{'params': model.ir_encoder.parameters()},{'params': model.vi_encoder.parameters()},
                            {'params': model.cross_encoder.parameters()},{'params': model.decoder.parameters()}],
                           lr=args.lr,betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'loss',
                                'loss_in',
                                'loss_grad',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

        train_log = train(args, train_loader, model, criterion_ssim_ir, criterion_gradient, criterion_perceptual,criterion_ssim_all, optimizer,writer,epoch)

        print('loss: %.4f - loss_ssim_ir_target: %.4f- loss_ssim_all: %.4f '
              % (train_log['loss'],
                 train_log['loss_ssim_ir_target'],
                 train_log['loss_ssim_all'],
                 ))

        tmp = pd.Series([
            epoch + 1,
            train_log['loss'],
            train_log['loss_ssim_ir_target'],
            train_log['loss_ssim_all'],
        ], index=['epoch', 'loss','loss_ssim_ir_target','loss_ssim_all'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch + 1) % args.name)


if __name__ == '__main__':
    main()
