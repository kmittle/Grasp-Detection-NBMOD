import os
import shutil

def copy_png_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in os.listdir(src_folder):
        if file_name.endswith(".png"):
            src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)
            shutil.copy(src_file_path, dest_file_path)
            print(f'复制文件: {src_file_path} 到 {dest_file_path}')

if __name__ == "__main__":
    source_folder = r'D:\Fruit_rd\img\a_bunch_of_bananas'
    destination_folder = r'J:\data1-RGB\img\a_bunch_of_bananas'
    
    copy_png_files(source_folder, destination_folder)
