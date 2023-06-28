import torch
from torch import nn
from model import get_model
import glob
import os
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
])


def cosine_similarity(v, M):
    # 计算向量v的模
    v_norm = np.linalg.norm(v)

    # 计算矩阵M每一列的模
    M_norm = np.linalg.norm(M, axis=0)

    # 计算向量v和矩阵M每一列的点积
    dot_product = np.dot(v, M)

    # 计算余弦相似度
    similarity = dot_product / (v_norm * M_norm)

    return similarity


if __name__ == '__main__':
    weights_path = 'weights/epoch6_loss_8.045684943666645.pth'
    img_dir = r'J:\experiment_data\0.1 test\test_img'
    target_img_index = 500

    img_path = glob.glob(img_dir + os.sep + '*.png')
    model = get_model()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    vectors = []
    for i in img_path:
        print(i)
        img = cv2.imread(i, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0)
        vector = model(img)
        vector = vector.squeeze().reshape(-1).detach().numpy()
        vector = vector.tolist()
        vectors.append(vector)
    vectors = np.array(vectors, dtype=np.float32).transpose()
    print(f'特征矩阵维度是：\n  {vectors.shape}')

    target_img = cv2.imread(img_path[target_img_index], -1)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_img = transform(target_img).unsqueeze(0)
    target_vector = model(target_img)
    target_vector = target_vector.squeeze().reshape(1, -1).detach().numpy()
    target_vector = target_vector.astype(np.float32)

    cos_similarity = cosine_similarity(target_vector, vectors)
    sorted_indices = np.argsort(-cos_similarity)
    v_sorted = np.take(cos_similarity, sorted_indices)
    # print(f'相似度向量：\n  {v_sorted}')
    # print(f'相似度序号向量：\n  {sorted_indices}')
    print(f'排序向量维度：\n  {sorted_indices.shape}')

    print(f'前10的相似度：\n  {v_sorted[0, :10]}')
    print(f'前10的图像：\n  {sorted_indices[0, :10]}')

    print(f'最后10个的相似度：\n  {v_sorted[0, -10:]}')
    print(f'最后10个的图像：\n  {sorted_indices[0, -10:]}')


