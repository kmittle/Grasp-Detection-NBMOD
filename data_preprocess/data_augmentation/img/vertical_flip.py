import os
import cv2


def vertical_flip(image):
    return cv2.flip(image, 0)


def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for file_name in file_list:
        input_image_path = os.path.join(input_folder, file_name)
        output_image_path = os.path.join(output_folder, file_name)

        image = cv2.imread(input_image_path)
        flipped_image = vertical_flip(image)

        cv2.imwrite(output_image_path, flipped_image)


if __name__ == "__main__":
    input_folder = r'J:\experiment_data\1 origin\img'
    output_folder = r'J:\experiment_data\5 vertical\img'

    process_images(input_folder, output_folder)
