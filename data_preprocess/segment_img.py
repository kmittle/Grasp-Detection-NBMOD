import cv2
import numpy as np


def partition_img(img_dir, img_name):
    img_path = img_dir + '\\' + img_name
    img = cv2.imread(img_path, -1)
    print(img.shape)
    h, w, _ = img.shape
    win_w, win_h = w//4, h//4
    k = 1
    for i in range(4):
        for j in range(4):
            sub_img = img[win_h*i:win_h*(i+1), win_w*j:win_w*(j+1), :]
            cv2.imwrite(img_dir + '\\' + str(k) + '.png', sub_img)
            k = k + 1


if __name__ == '__main__':
    partition_img(r'C:\Users\CBY\Desktop\manuscript\picture', 'jige.png')
