import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from dataloader import dataset
import tqdm
from model import ColorizationModel
import cv2


use_gpu = True


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


def test(loader_test, model_test):
    model_test.eval()

    for i, data_test in enumerate(tqdm.tqdm(loader_test)):
        if use_gpu:
            l = data_test['l'].cuda()
            hint = data_test['hint'].cuda()
            # l_inc = data_test['l_inc'].cuda()

        img_hint = torch.cat((l, hint), dim=1)

        output = model_test(img_hint)

        output_np = tensor2im(output)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_LAB2BGR)

        hint_np = tensor2im(img_hint)
        hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)

        file_name = data_test['file_name'][0]
        cv2.imwrite("outputs/Hint/" + str(i) + "hint.png", hint_bgr)
        cv2.imwrite("outputs/Output/" + file_name, output_bgr)


test_root = './cv_project/test_dataset'
test_dataset = dataset.ColorHintDataset(test_root, 256, "test")
test_dataloader = data.DataLoader(test_dataset, batch_size=1)

model_pretrained = ColorizationModel()
criterion = nn.L1Loss()

device = torch.device('cuda')
model_pretrained.to(device)
criterion.to(device)

checkpoint = torch.load("./checkpoints/model-epoch-39-losses-0.030.pth", map_location=device)
model_pretrained.load_state_dict(checkpoint['model'])
with torch.no_grad():
    test(test_dataloader, model_pretrained)
