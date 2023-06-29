import cv2
from PIL import Image
import numpy as np


def draw_function(picture_path, save_dir, model, transform):
    for i in picture_path:
        # img = cv2.imread(i)
        img = Image.open(i)
        img = img.convert('RGB')
        img1 = img.copy()
        img1 = np.array(img1)
        # img = cv2.resize(img,(400,300))
        img = transform(img)
        img = img.unsqueeze(dim=0)
        # img = img/255
        # img = img.reshape((1,300,400,3))

        # out1, out2, out3, out4, out5 = model.predict(img)
        predict_grasp = model(img)
        x = predict_grasp[1].item()  # x取第一个预测值
        y = predict_grasp[2].item()  # y取第二个预测值
        w = predict_grasp[3].item()  # w取第三个预测值
        h = predict_grasp[4].item()  # h取第四个预测值
        theta = predict_grasp[5].item()  # theta取第五个预测值
        center = (x, y)
        size = (w, h)
        angle = theta
        box = cv2.boxPoints((center, size, angle))
        box = np.int64(box)
        # predict_grasp = predict_grasp.cpu().detach().numpy()
        # vertice = np.zeros((4, 2))
        # x = predict_grasp[1]
        # y = predict_grasp[2]
        # w = predict_grasp[3]
        # h = predict_grasp[4]
        # theta = predict_grasp[5] / 180 * 3.1415927
        # vertice[0][0] = x - w / 2 * np.cos(theta) + h / 2 * np.sin(theta)
        # vertice[0][1] = y - w / 2 * np.sin(theta) - h / 2 * np.cos(theta)
        # vertice[1][0] = x + w / 2 * np.cos(theta) + h / 2 * np.sin(theta)
        # vertice[1][1] = y + w / 2 * np.sin(theta) - h / 2 * np.cos(theta)
        # vertice[2][0] = x + w / 2 * np.cos(theta) - h / 2 * np.sin(theta)
        # vertice[2][1] = y + w / 2 * np.sin(theta) + h / 2 * np.cos(theta)
        # vertice[3][0] = x - w / 2 * np.cos(theta) - h / 2 * np.sin(theta)
        # vertice[3][1] = y - w / 2 * np.sin(theta) + h / 2 * np.cos(theta)
        # p1 = (int(vertice[0][0]), int(vertice[0][1]))
        # p2 = (int(vertice[1][0]), int(vertice[1][1]))
        # p3 = (int(vertice[2][0]), int(vertice[2][1]))
        # p4 = (int(vertice[3][0]), int(vertice[3][1]))

        point_color1 = (255, 255, 0)  # BGR
        point_color2 = (255, 0, 255)  # BGR
        # point_color1 = (255, 255, 0)  # BGR
        # point_color2 = (255, 255, 0)  # BGR

        thickness = 2
        lineType = 4
        # img_p = k.numpy()
        # img = img.reshape((300,400,3))
        img_p = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img_p = img1

        cv2.line(img_p, box[0], box[3], point_color1, thickness, lineType)
        cv2.line(img_p, box[3], box[2], point_color2, thickness, lineType)
        cv2.line(img_p, box[2], box[1], point_color1, thickness, lineType)
        cv2.line(img_p, box[1], box[0], point_color2, thickness, lineType)

        picture_name = i.split('\\')[-1]
        save_path = save_dir + '\\' + picture_name
        cv2.imwrite(save_path, img_p)

