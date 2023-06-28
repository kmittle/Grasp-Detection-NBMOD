import cv2
import numpy as np
from shapely.geometry import Polygon
import glob
import torch
from torch import nn
import torchvision
from PIL import Image
from grasp_detect_singlebox import *


# 图像预处理方式
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((300, 400))
])


# 五维抓取坐标转四点坐标
def grasp_to_point(predict_grasp, radian=False):  # 输入抓取框的五维表示
    # vertice = np.zeros((4, 2))  # 生成一个4*2的数组用于保存四个点的八个坐标值
    x = predict_grasp[0].item()  # x取第一个预测值
    y = predict_grasp[1].item()  # y取第二个预测值
    w = predict_grasp[2].item()  # w取第三个预测值
    h = predict_grasp[3].item()  # h取第四个预测值
    theta = predict_grasp[4].item()  # theta取第五个预测值
    center = (x, y)
    size = (w, h)
    if radian:
        angle = theta / 3.1415927 * 180
    else:
        angle = theta
    box = cv2.boxPoints((center, size, angle))

    return box


# 计算jaccard指数
def intersection(g, p):  # 输入标签的四点坐标和预测的四点坐标
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


# 判断单个框与单个框之间是否抓取有效
def judge_availabel(predict_grasp, ground_truth):  # 输入五维抓取表示：预测的、标签的。       有效返回1，无效返回0
    predict_point = grasp_to_point(predict_grasp)  # 预测的五维抓取转四点坐标
    ground_truth_point = grasp_to_point(ground_truth, radian=True)  # 标签的五维抓取转四点坐标
    jaccard = intersection(ground_truth_point, predict_point)  # 计算二者的jaccard指数
    theta_predict = predict_grasp[-1].data.item()  # 取出预测的角度值
    theta_ground_truth = ground_truth[-1] / 3.1415927 * 180  # 取出标签的角度值
    
    # 以下代码将预测角度和Ground Truth转化到0-180度之间
    if theta_predict >= 180:
        theta_predict -= 180
    if theta_ground_truth >= 180:
        theta_ground_truth -= 180
    if theta_predict < 0:
        theta_predict += 180
    if theta_ground_truth < 0:
        theta_ground_truth += 180
    # 判定1
    distance_of_theta1 = abs(theta_predict - theta_ground_truth)
    # 以下代码将角度转化到-pi/2到+pi/2之间
    if theta_predict > 90:
        theta_predict -= 180
    if theta_ground_truth > 90:
        theta_ground_truth -= 180
    # 判定2
    distance_of_theta2 = abs(theta_predict - theta_ground_truth)
    # 综合判定
    distance_of_theta = min(distance_of_theta1, distance_of_theta2)  # 计算角度差

    if jaccard >= 0.25 and distance_of_theta <= 30:  # 符合有效抓取的条件
        available = 1
    else:
        available = 0
    return available


# 判断一张图是否抓取有效
def judge_picture(picture_path, text_path):   # 图片地址，标签地址。       有效返回1，无效返回0
    img = Image.open(picture_path)  # 读入单张要预测的图片
    img = img.convert('RGB')
    img = transform(img)
    img = img.unsqueeze(dim=0)
    img = img.to(device)
    predict_grasp = inference_single_image(img)  # 预测抓取位置的五维表示
    predict_grasp = predict_grasp[1:]
    # predict_grasp = predict_grasp.cpu().detach().numpy()
    # print(predict_grasp)
    # print(predict_grasp[0].detach().numpy())
    ground_truth = np.loadtxt(text_path)  # 读入标签文件
    flag = 0  # 标志位置0
    for i in range(len(ground_truth)):  # 遍历每一个标签中的抓取位置
        if judge_availabel(predict_grasp, ground_truth[i]) == 1:
            flag = 1
            break
    return flag


# 计算正确率
def evaluate_grasp(picture_dir_path, text_dir_path):  # 输入图片文件夹路径，标签文件夹路径
    text_path_s = glob.glob(text_dir_path + '\\' + '*.txt')  # 获取全部标签文件的路径
    text_path_s.sort(key=lambda x: x.split('\\')[-1].split('.txt')[0])  # 根据文件名进行排序
    img_path_s = glob.glob(picture_dir_path + '\\' + '*.png')  # 获取全部图片文件的路径
    img_path_s.sort(key=lambda x: x.split('\\')[-1].split('.png')[0])  # 根据文件名进行排序
    yes = 0
    total = 0
    for i in range(len(text_path_s)):
        available = judge_picture(img_path_s[i], text_path_s[i])  # 判断该图是否有效检测出有效抓取
        if available == 1:
            yes = yes + 1
            total = total + 1
            # print(img_path_s[i][-9:]+':Right')  #输出该图片检测正确的信息
        else:
            print(img_path_s[i].split('\\')[-1] + ':False')  # 输出该图片检测错误的信息
            total = total + 1
    print('检测总图片数：'+str(total))
    print('检测有效抓取数：'+str(yes))
    print('准确率：', yes/total)
    return yes / total


if __name__ == '__main__':
    # 权重文件路径 & 测试图片、标签文件夹地址
    weights_path = r'weights\epoch6_loss_8.045684943666645.pth'

    picture_dir_path = r'J:\experiment_data\0.1 test\single-simple\img'
    text_dir_path = r'J:\experiment_data\0.1 test\single-simple\label'

    # 指定测试评价设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 是否多卡训练
    multi_GPU = False

    # 定义模型
    inference_single_image = DetectSingleImage(device=device, weights_path=weights_path)

    # 测试模型
    evaluate_grasp(picture_dir_path, text_dir_path)

