import numpy as np
import torch
from torch import nn
from model import get_model
from Anchor import *
import cv2
from torchvision import transforms
import torch.nn.functional as F


class DetectMultiImage(nn.Module):
    def __init__(self, device, weights_path, multi_gpu=False, weights=True):
        super(DetectMultiImage, self).__init__()
        self.net = get_model().to(device)
        if multi_gpu:
            self.net = nn.DataParallel(self.net)
        if weights:
            self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()

    def get_index_and_bias(self, output, confidence_threshold):
        N, C, H, W = output.shape
        # N 90 26 26 ----> N 26 26 90
        output = output.permute(0, 2, 3, 1)
        # N 26 26 90 ----> N 26 26 15 6
        output = output.reshape(N, H, W, num_anchors, -1)
        # 取出置信度大于设定阈值confidence_threshold的所有box, mask_obj的shape：N W H num_anchors
        mask_obj = (F.sigmoid(output[..., 0]) >= confidence_threshold)
        # 返回mask_obj中为True的坐标索引，index的shape：为True的个数 x 4  （4的含义：N H W num_anchors的索引）
        index = mask_obj.nonzero()
        # 获得偏移量：confidence，tx，ty，tw，th，t_theta，bias的shape：为True的个数 x 6  （6的含义：box的6个属性值）
        bias = output[mask_obj]
        return index, bias

    def get_coordinate(self, index, bias):
        # 以下，confidence, cx, cy, w, h, theta的shape都是：为True的个数 x 1
        confidence = torch.sigmoid(bias[:, 0])
        # cx = index * field + sigmoid(bias) * filed
        cx = (index[:, 2] + torch.sigmoid(bias[:, 1])) * field_of_grid_cell
        cy = (index[:, 1] + torch.sigmoid(bias[:, 2])) * field_of_grid_cell
        w = anchor_w * torch.exp(bias[:, 3])
        h = anchor_h * torch.exp(bias[:, 4])
        # theta已经转换为角度制
        theta = (index[:, 3] + torch.sigmoid(bias[:, 5])) * theta_margin
        return confidence, cx, cy, w, h, theta

    def forward(self, input, confidence_threshold):
        output = self.net(input)
        index, bias = self.get_index_and_bias(output, confidence_threshold)
        confidence, cx, cy, w, h, theta = self.get_coordinate(index, bias)
        # 返回shape：为True的个数 x 6
        return torch.cat([confidence.unsqueeze(1),
                          cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1), theta.unsqueeze(1)], dim=1)


def draw_multi_box(img, box_coordinates):
    for i in range(box_coordinates.shape[0]):
        center = (box_coordinates[i, 1].item(), box_coordinates[i, 2].item())
        size = (box_coordinates[i, 3].item(), box_coordinates[i, 4].item())
        angle = box_coordinates[i, 5].item()
        box = cv2.boxPoints((center, size, angle))
        box = np.int64(box)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weights_path = 'weights/epoch6_loss_8.045684943666645.pth'

    img = cv2.imread(r'J:\experiment_data\MOS\img\000000r.png')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    inference_multi_image = DetectMultiImage(device=device, weights_path=weights_path)

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = transform(img2).unsqueeze(dim=0).to(device)

    boxes = inference_multi_image(img2, 0.9999)

    print(boxes.shape)
    print(boxes[:, 0].data[:5])

    draw_multi_box(img, boxes.data)

