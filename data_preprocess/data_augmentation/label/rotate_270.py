import os
import os.path
import xml.etree.ElementTree as ET
import glob
import numpy as np


def center_to_vertice(x, y, w, h, angle):  # 将抓取参数转化为抓取框用以显示
    theta = angle
    vertice = np.zeros((4, 2))
    vertice[0] = (x - w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) - h / 2 * np.cos(theta))
    vertice[1] = (x + w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) - h / 2 * np.cos(theta))
    vertice[2] = (x + w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
    vertice[3] = (x - w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
    for i in range(0, 2):
        for j in range(0, 4):
            vertice[j][i] = round(vertice[j][i], 3)
    return vertice


def xml_to_txt(xml_path, txt_path):
    os.chdir(xml_path)  # 将当前的工作目录转到该目录下
    annotations = os.listdir('.')  # 返回指定路径下的文件和文件夹列表
    annotations = glob.glob(str(annotations) + '*.xml')  # 返回所有匹配的文件路径列表
    fileList = os.listdir(xml_path)
    print(fileList)
    k = 0
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    for i, file in enumerate(annotations):
        in_file = open(file)  # 打开xml文件
        tree = ET.parse(in_file)  # 用ElementTree表示xml文件
        root = tree.getroot()  # 返回树的根节点
        # filename = root.find('filename').text
        # s = filename[2:]
        # s = s.zfill(5)
        # file_save = s + '.txt'
        # file_save = filename + '.txt'
        file_save = fileList[k][:-4] + '.txt'
        k = k + 1
        print(file_save)
        file_txt = os.path.join(txt_path, file_save)
        f_w = open(file_txt, 'w')
        for obj2 in root.iter('object'):
            # current = list()
            # class_num = class_names.index(name)
            xmlbox1 = obj2.find('robndbox')
            x = xmlbox1.find('cx').text
            y = xmlbox1.find('cy').text
            width = xmlbox1.find('w').text
            height = xmlbox1.find('h').text
            angle = xmlbox1.find('angle').text
            x = float(x)
            x = x * 0.65
            y = float(y)
            y = y * 0.65
            width = float(width)
            width = width * 0.65
            height = float(height)
            height = height * 0.65
            angle = float(angle)
            if height > width:
                exchange = width
                width = height
                height = exchange
                angle = angle + 3.1415926/2
                if angle >= 3.1415926:
                    angle = angle - 3.1415926

            exchange = x
            x = y
            y = 416 - exchange  # 顺时针旋转270度
            angle = angle + 3.1415926 * 1.5
            while angle >= 3.1415926:
                angle = angle - 3.1415926

            # angle = np.tan(angle)
            # if angle<0:
            #    angle = max(-15,angle)
            # if angle>=0:
            #    angle = min(15,angle)
            
            f_w.write(str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' ' + str(angle) + '\n')
            
            # f_w.write(str(bbox[0][0]) + ' ' + str(bbox[0][1]) + '\n' +
            #           str(bbox[1][0]) + ' ' + str(bbox[1][1]) + '\n' +
            #           str(bbox[2][0]) + ' ' + str(bbox[2][1]) + '\n' +
            #           str(bbox[3][0]) + ' ' + str(bbox[3][1]) + '\n')


if __name__ == "__main__":
    xml_path = r'J:\experiment_data\0 train_test_split\train\label'  # xml文件路径
    txt_path = r'J:\experiment_data\8 r270\label'  # txt文件路径

    os.chdir(xml_path)  # 将当前的工作目录转到该目录下
    annotations = os.listdir('.')  # 返回指定路径下的文件和文件夹列表
    annotations = glob.glob(str(annotations) + '*.xml')  # 返回所有匹配的文件路径列表
    # for i, element in enumerate(annotations):
    #     print(i, element)

    xml_to_txt(xml_path, txt_path)
