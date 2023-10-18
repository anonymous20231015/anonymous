import os
import re

def get_root_path():
    # 获取当前目录
    cwd = os.getcwd()
    if cwd.find("/") == -1:
        cwd += "\\"
        root_name = re.search(r"(Fed.*?\\)", cwd).group(1)
    else:
        cwd += "/"
        root_name = re.search(r"(Fed.*?/)", cwd).group(1)
    # 获取根目录
    root_path = cwd[:cwd.find(root_name) + len(root_name)]

    return root_path

