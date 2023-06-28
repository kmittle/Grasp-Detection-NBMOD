import os
import shutil


def copy_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in os.listdir(src_folder):
        print(file_name)
        src_file_path = os.path.join(src_folder, file_name)
        dest_file_path = os.path.join(dest_folder, file_name)

        if os.path.isfile(src_file_path):
            shutil.copy2(src_file_path, dest_file_path)


if __name__ == "__main__":
    src_folder = r"J:\experiment_data\8 r270\label"  # 替换为源文件夹的路径
    dest_folder = r"J:\experiment_data\Train_Augmented_data\label"  # 替换为目标文件夹的路径

    copy_files(src_folder, dest_folder)
