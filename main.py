import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import dataset
from dataloader import transform as tf
import torchvision.models as models
import cv2


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)


class ColorizationModel(nn.Module):
    def __int__(self, mid_input_size=128, global_input_size=512):
        super(ColorizationModel, self).__init__()

        # encoder
        resnet = models.resnet18()
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1).data)
        self.mid_level_resnet = nn.Sequential(*list(resnet.children())[0:6])

        # decoder
        self.deconv1_new = nn.ConvTranspose2d(mid_input_size, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(mid_input_size, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.mid_level_resnet(x)

        x = nn.ReLU(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = nn.ReLU(self.bn3(self.conv2(x)))
        x = nn.ReLU(self.conv3(x))
        x = self.upsample(x)
        x = nn.ReLU(self.conv4(x))
        x = self.upsample(self.conv5(x))

        return x


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(loader_train, model_train, crit, opt, epoch):
    print('Starting training epoch {}'.format(epoch + 1))
    model_train.train()

    batch_time, data_time, set_loss = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()

    for i, component in enumerate(loader_train):
        if use_gpu:
            l = component["l"].cuda()
            ab = component["ab"].cuda()
            hint = component["hint"].cuda()
            mask = component["mask"].cuda()

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint, mask), dim=1)
        data_time.update(time.time() - start)

        output_hint = model_train(hint_image)
        loss = crit(output_hint, gt_image)
        set_loss.update(loss.item(), hint_image.size(0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % 225 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch + 1, i, len(loader_train), batch_time=batch_time,
                    data_time=data_time, loss=set_loss))

    print('Finished training epoch {}'.format(epoch + 1))


def validate(loader_val, model_val, crit):
    model_val.eval()

    batch_time, data_time, set_loss = AverageMeter(), AverageMeter(), AverageMeter()
    start = time.time()

    for i, component in enumerate(loader_val):
        if use_gpu:
            l = component["l"].cuda()
            ab = component["ab"].cuda()
            hint = component["hint"].cuda()
            mask = component["mask"].cuda()

        gt_image = torch.cat((l, ab), dim=1)
        hint_image = torch.cat((l, hint, mask), dim=1)
        data_time.update(time.time() - start)
        output_hint = model_val(hint_image)

        loss = crit(output_hint, gt_image)
        set_loss.update(loss.item(), hint_image.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        if i % 100 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(loader_val), batch_time=batch_time, loss=set_loss))
        out_hint_np = tensor2im(output_hint)
        out_hint_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        cv2.imwrite("outputs/outputs/output_" + str(i) + ".png", out_hint_bgr)

    print('Finished validation.')
    return set_loss.avg


use_gpu = torch.cuda.is_available()

if __name__ == "__main__":
    root_path = './cv_project'
    train_root = './cv_project/train'
    val_root = './cv_project/val'
    test_root = './cv_project/test_dataset'

    train_dataset = dataset.ColorHintDataset(root_path, 256, "train")
    train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = dataset.ColorHintDataset(root_path, 256, "val")
    val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    test_dataset = dataset.ColorHintDataset(test_root, 256, "test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = ColorizationModel()
    criterion = nn.MSELoss()

    save_images = True
    best_losses = 1e10
    epochs = 100

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-2, weight_decay=0.0)

    for e in range(epochs):
        train(train_dataloader, model, criterion, optimizer, e)
        with torch.no_grad():
            losses = validate(val_dataloader, model, criterion)
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(e + 1, losses))
