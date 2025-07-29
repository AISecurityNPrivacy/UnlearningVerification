import os
import re

# 当前目录（你也可以指定一个路径）
root_path = "."

# 用于匹配以 BBCNews_SIM_ 开头，后面是一个浮点数字的文件夹
pattern = re.compile(r"^BBCNews_SIM_(\d*\.?\d+)$")

for folder in os.listdir(root_path):
    match = pattern.match(folder)
    if match:
        num_str = match.group(1)
        try:
            # 将提取的字符串转为 float，再保留两位小数，构造新目录名
            formatted = f"{float(num_str):.2f}"
            new_name = f"BBCNews_SIM_{formatted}"
            old_path = os.path.join(root_path, folder)
            new_path = os.path.join(root_path, new_name)

            # 重命名（如果新名字不同且目标不存在）
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {folder} -> {new_name}")
        except ValueError:
            print(f"Failed to parse number from: {folder}")