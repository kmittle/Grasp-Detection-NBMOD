import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from to_yolo_dataset import *
from model import get_model
from Anchor import *
from torch.utils.tensorboard import SummaryWriter
import glob


def loss_fn(output, label):
    N, C, H, W = output.shape
    # N C H W ----> N H W C
    output = output.permute(0, 2, 3, 1)
    # N H W C ----> N H W num_anchors 6
    output = output.reshape(N, H, W, num_anchors, -1)
    # mask_obj：掩码张量，取出有Ground Truth匹配的Anchor；[..., dim]：取出最后一个维度，这里取出符合条件的anchor的置信度
    mask_obj = label[..., 0] == 1
    # 无物体的掩码张量
    mask_no_obj = label[..., 0] == 0

    loss_confidence_fn = nn.BCEWithLogitsLoss()
    # 计算置信度正负样本都要用上，该损失函数网络输出在前，标签值在后
    loss_confidence_positive = loss_confidence_fn(output[mask_obj][..., 0], label[mask_obj][..., 0])
    loss_confidence_negative = loss_confidence_fn(output[mask_no_obj][..., 0], label[mask_no_obj][..., 0])

    loss_regression_fn = nn.MSELoss()
    # Rotation Box回归损失，x, y, w, h, theta
    loss_box = loss_regression_fn(output[mask_obj][..., 1:6], label[mask_obj][..., 1:6])

    loss = 2 * loss_confidence_positive + 0.008 * loss_confidence_negative + 10 * loss_box
    return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 是否使用预训练权重
    pretrained = True

    # 设置Batch Size
    Batchsize = 4

    # 设置数据集路径
    train_img_path = glob.glob(r'data\train_data\img\*.png')
    train_label_path = glob.glob(r'data\train_data\label\*.txt')
    test_img_path = r''
    test_label_path = r''
    print('总样本数：', len(train_img_path))

    train_dataset = YoloDataset(train_img_path, train_label_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Batchsize,
        shuffle=True
    )

    net = get_model().to(device)
    # net_weights = net.state_dict()
    # 加载预训练权重
    if pretrained:
        pretrained_weights_path = r'pretrained_weights\mobilevit_s.pt'
        pretrained_weights = torch.load(pretrained_weights_path,
                                        # map_location=device
                                        )
        del_key = []
        for key, _ in pretrained_weights.items():
            if ('fc' in key) or ('global_pool' in key) or ('flatten' in key) or ('dropout' in key) or ('conv_1x1_exp' in key):
                del_key.append(key)

        for key in del_key:
            del pretrained_weights[key]

        # missing_keys：net中有而预训练权重中没有的键值对；unexpected_keys：预训练权重中有而net中没有的键值对
        missing_keys, unexpected_keys = net.load_state_dict(pretrained_weights, strict=False)
        print('missing_keys：')
        print(missing_keys)
        print('unexpected_keys：')
        print(unexpected_keys)
        print('预训练权重加载完成')

    # 指定优化器
    opt = optim.AdamW(net.parameters())

    # Epoch数
    Epochs = 30

    # 训练日志
    writer = SummaryWriter(r'log')

    # 训练循环
    print('开始训练：')
    for epoch in range(1, Epochs + 1):
        epoch_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.data.item()

        epoch_loss /= len(train_dataloader)

        writer.add_scalar('train_loss', epoch_loss, epoch)

        torch.save(net.state_dict(), 'weights/epoch{}_loss_{}.pth'.format(epoch, epoch_loss))

        # if epoch <= 500:
        #     if epoch % 50 == 0:
        #         torch.save(net.state_dict(), 'weights/epoch{}_loss_{}.pth'.format(epoch, epoch_loss))
        # if 500 < epoch <= 950:
        #     if epoch % 20 == 0:
        #         torch.save(net.state_dict(), 'weights/epoch{}_loss_{}.pth'.format(epoch, epoch_loss))
        # if epoch > 950:
        #     torch.save(net.state_dict(), 'weights/epoch{}_loss_{}.pth'.format(epoch, epoch_loss))

        print('Epoch{}_loss: {}'.format(epoch, epoch_loss))
