import os
import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, sigma=25):
    height, width, channels = image.shape
    noise = np.random.normal(mean, sigma, (height, width, channels))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_noise_to_folder(input_folder, output_folder, mean=0, sigma=25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for file_name in file_list:
        input_image_path = os.path.join(input_folder, file_name)
        output_image_path = os.path.join(output_folder, file_name)

        image = cv2.imread(input_image_path)
        noisy_image = add_gaussian_noise(image, mean, sigma)

        cv2.imwrite(output_image_path, noisy_image)


if __name__ == "__main__":
    input_folder = r'J:\experiment_data\0 train_test_split\train\img'
    output_folder = r'J:\experiment_data\2 Gs_noise\img'
    mean = 0
    sigma = 25  # 根据需求调整噪声强度

    add_noise_to_folder(input_folder, output_folder, mean, sigma)
