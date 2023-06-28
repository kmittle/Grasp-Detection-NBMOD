import numpy as np
import torch
from torch import nn
from model import get_model
from Anchor import *
import cv2
from torchvision import transforms


class DetectSingleImage(nn.Module):
    def __init__(self, device, weights_path, multi_gpu=False, weights=True):
        super(DetectSingleImage, self).__init__()
        self.net = get_model().to(device)
        if multi_gpu:
            self.net = nn.DataParallel(self.net)
        if weights:
            self.net.load_state_dict(torch.load(weights_path))
            print('载入权重完成')
        self.net.eval()

    def get_index_and_bias(self, output):
        N, C, H, W = output.shape
        # N C H W ----> N H W C
        output = output.permute(0, 2, 3, 1)
        # N H W C ----> N H W num_anchors 6
        output = output.reshape(N, H, W, num_anchors, -1)
        # 只取出置信度最大的box
        select_box = torch.max(output[..., 0])
        # mask_obj的shape：N H W num_anchors，只取出置信度最大的box
        mask_obj = (output[..., 0] == select_box)
        # 返回mask_obj中为True的坐标索引，这里指定第一个元素，因为可能出现多个box置信度相同且都是最大的情况，index：N H W num_anchors
        index = mask_obj.nonzero()[0]
        # index = mask_obj.nonzero()
        # 获得偏移量：confidence，tx，ty，tw，th，t_theta
        # bias = output[index[0]][index[1]][index[2]][index[3]]
        bias = output[mask_obj][0]
        return index, bias

    def get_coordinate(self, index, bias):
        confidence = torch.sigmoid(bias[0])
        cx = (index[2] + torch.sigmoid(bias[1])) * field_of_grid_cell
        cy = (index[1] + torch.sigmoid(bias[2])) * field_of_grid_cell
        w = anchor_w * torch.exp(bias[3])
        h = anchor_h * torch.exp(bias[4])
        theta = (index[3] + torch.sigmoid(bias[5])) * theta_margin
        return confidence, cx, cy, w, h, theta

    def forward(self, input):
        output = self.net(input)
        index, bias = self.get_index_and_bias(output)
        confidence, cx, cy, w, h, theta = self.get_coordinate(index, bias)
        return torch.cat([confidence.unsqueeze(0),
                          cx.unsqueeze(0), cy.unsqueeze(0), w.unsqueeze(0), h.unsqueeze(0), theta.unsqueeze(0)], dim=0)


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


if __name__ == '__main__':
    weights_path = r'weights\Feature_Concat\epoch12_loss_424.52915453940744.pth'

    img = cv2.imread(r'J:\experiment_data\0.1 test\single-complex\img\000009r.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    inference_single_image = DetectSingleImage(device=device, weights_path=weights_path)

    # img = np.random.randn(416, 416, 3).astype(np.float32)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = transform(img2).unsqueeze(dim=0).to(device)

    box = inference_single_image(img2)
    print(box.shape)
    print('置信度：', box[0].data.item())

    draw_one_box(img, box)

    # draw_one_box(img,
    #              confidence.data.item(),
    #              cx.data.item(), cy.data.item(), w.data.item(), h.data.item(), theta.data.item())
