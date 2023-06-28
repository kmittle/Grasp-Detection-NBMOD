import glob

import numpy as np
import torch
from torch import nn
from model import mobile_vit_small
from Anchor import *
import cv2
from torchvision import transforms
from grasp_detect_singlebox import DetectSingleImage
from draw_function import draw_function


def draw_one_box(img, coordinate):
    # center = (cx, cy)
    # size = (w, h)
    # angle = theta
    center = (coordinate[1].item(), coordinate[2].item())
    size = (coordinate[3].item(), coordinate[4].item())
    angle = coordinate[5].item()
    box = cv2.boxPoints((center, size, angle))
    box = np.int64(box)
    # print(box)
    # Font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'c: ' + str(round(coordinate[0].item(), 3)), (box[3][0], box[3][1]), Font, 0.5, (0, 0, 255), 1)
    cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_pictures(imgs_path, save_dir):
    for i in imgs_path:
        img = cv2.imread(i)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = transform(img2).unsqueeze(dim=0).to(device)
        box = inference_single_image(img2)
        center = (box[1].item(), box[2].item())
        size = (box[3].item(), box[4].item())
        angle = box[5].item()
        box = cv2.boxPoints((center, size, angle))
        box = np.int64(box)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        cv2.imwrite(save_dir + '\\' + i.split('\\')[-1], img)


if __name__ == '__main__':
    # 权重路径
    weights_path = r'weights\epoch6_loss_8.045684943666645.pth'

    # 图像文件夹路径
    imgs_path = glob.glob(r'J:\experiment_data\0.1 test\single-complex\img\*.png')

    # 保存路径
    save_dir = r'detected_img\complex'

    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    inference_single_image = DetectSingleImage(device=device, weights_path=weights_path)

    draw_function(picture_path=imgs_path, save_dir=save_dir, model=inference_single_image, transform=transform)

    # draw_pictures(imgs_path=imgs_path, save_dir=save_dir)

    # print('置信度：', box[0].data.item())

    # draw_one_box(img,
    #              confidence.data.item(),
    #              cx.data.item(), cy.data.item(), w.data.item(), h.data.item(), theta.data.item())

