import os
import cv2
import numpy as np
from numpy import linalg
import random
from PIL import Image
from sklearn.decomposition import PCA


def pca_color_augmentation(image_array):
    '''
    image augmention: PCA jitter
    :param image_array: 图像array
    :return img2: 经过PCA-jitter增强的图像array
    '''
    assert image_array.dtype == 'uint8'
    assert image_array.ndim == 3
    # 输入的图像应该是 (w, h, 3)这样的三通道分布

    img1 = image_array.astype('float32') / 255.0
    # 分别计算R，G，B三个通道的方差和均值
    mean = img1.mean(axis=0).mean(axis=0)
    std = img1.reshape((-1, 3)).std()  # 不可以使用img1.std(axis = 0).std(axis = 0)

    # 将图像标按channel标准化（均值为0，方差为1）
    img1 = (img1 - mean) / (std)

    # 将图像按照三个通道展成三个长条
    img1 = img1.reshape((-1, 3))

    # 对矩阵进行PCA操作
    # 求矩阵的协方差矩阵
    cov = np.cov(img1, rowvar=False)
    # 求协方差矩阵的特征值和向量
    eigValue, eigVector = linalg.eig(cov)

    # 抖动系数（均值为0，方差为0.1的标准分布）
    rand = np.array([random.normalvariate(0, 0.08) for i in range(3)])
    jitter = np.dot(eigVector, eigValue * rand)

    jitter = (jitter * 255).astype(np.int32)[np.newaxis, np.newaxis, :]

    img2 = np.clip(image_array + jitter, 0, 255)

    return img2


def process_images(input_folder, output_folder, alpha_std=0.1, seed=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for file_name in file_list:
        input_image_path = os.path.join(input_folder, file_name)
        output_image_path = os.path.join(output_folder, file_name)

        image = cv2.imread(input_image_path)
        augmented_image = pca_color_augmentation(image)

        cv2.imwrite(output_image_path, augmented_image)
        print(output_image_path)


if __name__ == "__main__":
    input_folder = r'J:\experiment_data\1 origin\img'
    output_folder = r'J:\experiment_data\3 PCA_illumination\img'
    alpha_std = 0.1  # 根据需求调整光照强度的变化程度
    seed = 42  # 可以修改为任意整数以改变随机种子，或设置为None以使用随机种子

    process_images(input_folder, output_folder, alpha_std, seed)
