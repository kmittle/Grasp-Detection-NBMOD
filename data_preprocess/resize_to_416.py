import os
import cv2


def pad_and_resize_image(image):
    # 创建一个黑色的640x640背景图像
    padded_image = cv2.copyMakeBorder(image, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # 缩放图像到416x416
    resized_image = cv2.resize(padded_image, (416, 416), interpolation=cv2.INTER_AREA)

    return resized_image


def process_images(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in os.listdir(src_folder):
        if file_name.endswith(".png"):
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)

            # 读取图像
            image = cv2.imread(src_file_path)

            # 调用 pad_and_resize_image 函数处理图像
            processed_image = pad_and_resize_image(image)

            # 保存处理后的图像
            cv2.imwrite(dest_file_path, processed_image)
            print(f'处理并保存图像: {src_file_path} 到 {dest_file_path}')


if __name__ == "__main__":
    source_folder = r'D:\cornell_data\img'
    destination_folder = r'J:\cornell_dataset\img'

    process_images(source_folder, destination_folder)
