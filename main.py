import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from dataloader import dataset
from dataloader import transform as tf
import tqdm
from model import ColorizationModel, AverageMeter
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


def train(loader_train, model_train, crit, opt, epoch):
    model_train.train()
    set_loss = AverageMeter()

    print('Starting training epoch {}'.format(epoch + 1))

    for i, data_t in enumerate(tqdm.tqdm(loader_train)):
        if use_gpu:
            l = data_t['l'].cuda()
            ab = data_t['ab'].cuda()
            hint = data_t['hint'].cuda()

        img_lab = torch.cat((l, ab), dim=1)
        img_hint = torch.cat((l, hint), dim=1)

        output = model_train(img_hint, )
        loss = crit(output, img_lab)
        set_loss.update(loss.item(), img_hint.size(0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch + 1, i,
                                                                                         len(loader_train),
                                                                                         loss=set_loss))

    print('Finished training epoch {}'.format(epoch + 1))


def validate(loader_val, model_val, crit):
    model_val.eval()
    set_loss = AverageMeter()

    for i, data_v in enumerate(tqdm.tqdm(loader_val)):
        if use_gpu:
            l = data_v['l'].cuda()
            ab = data_v['ab'].cuda()
            hint = data_v['hint'].cuda()

        img_lab = torch.cat((l, ab), dim=1)
        img_hint = torch.cat((l, hint), dim=1)

        output = model_val(img_hint)
        loss = crit(output, img_lab)
        set_loss.update(loss.item(), img_hint.size(0))

        if i % 100 == 0:
            print('Validate: [{0}/{1}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(loader_val),
                                                                                       loss=set_loss))

        output_np = tensor2im(output)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2BGR)

        hint_np = tensor2im(img_hint)
        hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)

        lab_np = tensor2im(img_lab)
        lab_bgr = cv2.cvtColor(lab_np, cv2.COLOR_LAB2BGR)

        psnr = cv2.PSNR(lab_bgr, output_bgr)

        cv2.imwrite("outputs/GroundTruth/" + str(i) + "lab.png", lab_bgr)
        cv2.imwrite("outputs/Hint/" + str(i) + "hint.png", hint_bgr)
        cv2.imwrite("outputs/Output/" + "_psnr:" + str(psnr) + ".png", output_bgr)

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

    best_losses = 1e10
    epochs = 100

    device = torch.device('cuda')
    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0012, weight_decay=1e-6)

    for e in range(epochs):
        train(train_dataloader, model, criterion, optimizer, e)
        with torch.no_grad():
            losses = validate(val_dataloader, model, criterion)
        if losses < best_losses:
            best_losses = losses
            PATH = './checkpoints'
            if os.path.isdir(PATH) is False:
                os.mkdir(PATH)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, PATH + '/model-epoch-{}-losses-{:.3f}.pth'.format(e + 1, losses))
