from torch.utils.data import Dataset
import numpy as np
from Anchor import anchor_w, anchor_h, theta_margin, Anchor_eps, num_anchors, num_grid_cell, field_of_grid_cell, anchor_thetas
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import DataLoader
import pandas as pd
import math


def get_one_label(label_path):
    # 每行标签的顺序是x, y, w, h, theta
    # 最终输出的下采样倍数是4倍，每个grid cell对应原图上16*16的位置
    labels = np.loadtxt(label_path)
    # labels = pd.read_csv(label_path, header=None, sep=' ').to_numpy()
    # tensor的每个box的标签值顺序是：confidence、bx、by、bw、bh、theta
    tensor = np.zeros((num_grid_cell, num_grid_cell, num_anchors, 6))
    for box in labels:
        x = int(box[0] // field_of_grid_cell)
        bx = (box[0] % field_of_grid_cell) / field_of_grid_cell
        # bx = sigmoid(tx)
        tx = math.log((bx + Anchor_eps) / (1 - bx))
        y = int(box[1] // field_of_grid_cell)
        by = (box[1] % field_of_grid_cell) / field_of_grid_cell
        # by = sigmoid(ty)
        ty = math.log((by + Anchor_eps) / (1 - by))
        bw = box[2] / anchor_w
        # bw = exp(tw)
        tw = math.log(bw + Anchor_eps)
        bh = box[3] / anchor_h
        # bh = exp(th)
        th = math.log(bh + Anchor_eps)
        # 这里theta是弧度制，需要转换为角度制
        while box[4] >= 3.1415927:
            box[4] -= 3.1415927
        theta = box[4] / 3.1415927 * 180
        theta_anchor_match = int(theta // theta_margin)
        b_theta = (theta % theta_margin) / theta_margin
        # b_theta = sigmoid(t_theta)
        t_theta = math.log((b_theta + Anchor_eps) / (1 - b_theta))
        # 赋值
        tensor[y][x][theta_anchor_match][0] = 1
        tensor[y][x][theta_anchor_match][1] = tx
        tensor[y][x][theta_anchor_match][2] = ty
        tensor[y][x][theta_anchor_match][3] = tw
        tensor[y][x][theta_anchor_match][4] = th
        tensor[y][x][theta_anchor_match][5] = t_theta

    tensor = tensor.astype(np.float32)
    return tensor


def get_label(label_path):
    labels = []
    for i in label_path:
        label = get_one_label(i)
        labels.append(label)
    return np.array(labels)


transform = transforms.Compose([
    transforms.ToTensor()
])


class YoloDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path

    def __getitem__(self, index):
        x = self.img_path[index]
        x = Image.open(x)
        x = x.convert('RGB')
        x = transform(x)
        y = self.label_path[index]
        y = get_one_label(y)
        return x, y

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    img_path = glob.glob(r'data\train_data\img\*.png')
    label_path = glob.glob(r'data\train_data\label\*.txt')

    dataset = YoloDataset(img_path, label_path)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )
    img, label = next(iter(dataloader))
    print(img.shape)
    print(label.shape)
    print(label[0][int(195.579085//32)][int(160.0963//32)])
    # 160.0963 195.579085
