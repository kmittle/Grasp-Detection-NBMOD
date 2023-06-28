import os


# 初始值
path = r'J:\experiment_data\8 r270\label'

start_num = 178501  # 命名起点


file_type = "png" if path[-3:] == 'img' else "txt"


# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)


n = 0
for i in fileList:
    
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]   # os.sep添加系统分隔符
    
    s = str(start_num)
    s = s.zfill(7)
    
    newname = path + os.sep + s + "r." + file_type
    
    os.rename(oldname, newname)   # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    start_num = start_num + 1
    n = n+1

