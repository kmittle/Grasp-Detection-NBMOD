import os
from PIL import Image


def rotate_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for file_name in file_list:
        print(file_name)
        src_image_path = os.path.join(input_folder, file_name)
        dest_image_path = os.path.join(output_folder, file_name)

        img = Image.open(src_image_path)
        rotated_img = img.rotate(-270)  # 顺时针旋转90度
        rotated_img.save(dest_image_path)


if __name__ == "__main__":
    input_folder = r'J:\experiment_data\1 origin\img'
    output_folder = r'J:\experiment_data\8 r270\img'

    rotate_images(input_folder, output_folder)
