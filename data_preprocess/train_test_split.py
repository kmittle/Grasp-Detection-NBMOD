import os
import shutil
import random


def split_dataset(image_folder, label_folder,
                  train_image_folder, train_label_folder, test_image_folder, test_label_folder, ratio, seed):
    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)

    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder)

    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)

    if not os.path.exists(test_label_folder):
        os.makedirs(test_label_folder)

    file_list = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    random.seed(seed)
    random.shuffle(file_list)

    train_size = int(len(file_list) * ratio)

    for idx, file_name in enumerate(file_list):
        src_image_path = os.path.join(image_folder, file_name)
        src_label_path = os.path.join(label_folder, file_name.replace(".png", ".xml"))

        if idx < train_size:
            dest_image_path = os.path.join(train_image_folder, file_name)
            dest_label_path = os.path.join(train_label_folder, file_name.replace(".png", ".xml"))
        else:
            dest_image_path = os.path.join(test_image_folder, file_name)
            dest_label_path = os.path.join(test_label_folder, file_name.replace(".png", ".xml"))

        shutil.copy(src_image_path, dest_image_path)
        shutil.copy(src_label_path, dest_label_path)


if __name__ == "__main__":
    images_folder = r'J:\data_resized_to_416\img\single-complex'
    labels_folder = r'J:\data_resized_to_416\label\single-complex'
    train_images_folder = r'J:\experiment_data\train_test_split\train\img\single-complex'
    train_labels_folder = r'J:\experiment_data\train_test_split\train\label\single-complex'
    test_images_folder = r'J:\experiment_data\train_test_split\test\img\single-complex'
    test_labels_folder = r'J:\experiment_data\train_test_split\test\label\single-complex'

    split_ratio = 1 - 500 / 13000  # 训练集所占的比例
    random_seed = 42  # 随机数种子

    split_dataset(images_folder, labels_folder,
                  train_images_folder, train_labels_folder, test_images_folder, test_labels_folder, split_ratio, random_seed)
