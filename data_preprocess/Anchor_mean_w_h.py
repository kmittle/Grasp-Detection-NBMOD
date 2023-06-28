import os
import numpy as np


def compute_column_mean(folder_path):
    column_sum_3 = 0
    column_sum_4 = 0
    total_rows = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = np.loadtxt(file_path)

            column_sum_3 += np.sum(data[:, 2])  # 第3列的和
            column_sum_4 += np.sum(data[:, 3])  # 第4列的和
            total_rows += data.shape[0]  # 累加行数

    mean_3 = column_sum_3 / total_rows
    mean_4 = column_sum_4 / total_rows

    return mean_3, mean_4


if __name__ == "__main__":
    folder_path = r"J:\experiment_data\1 origin\label"
    mean_3, mean_4 = compute_column_mean(folder_path)
    print("第3列的均值: ", mean_3)
    print("第4列的均值: ", mean_4)
