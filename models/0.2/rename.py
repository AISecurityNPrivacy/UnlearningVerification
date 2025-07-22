import os
import re

def is_pth_with_number(filename):
    """判断是否是以数字结尾的 .pth 文件"""
    return re.match(r".*?(\d+)\.pth$", filename)

def get_numeric_suffix(filename):
    """提取结尾数字"""
    match = re.match(r".*?(\d+)\.pth$", filename)
    return int(match.group(1)) if match else None

def process_directory(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 只处理最小子目录（不再往下递归）
        if not dirnames:
            pth_files = [f for f in filenames if f.endswith(".pth")]
            numbered_files = [f for f in pth_files if is_pth_with_number(f)]
            non_numbered_files = [f for f in pth_files if not is_pth_with_number(f)]

            # 如果存在非数字结尾的 .pth 文件，跳过该目录
            if non_numbered_files:
                continue

            if not numbered_files:
                continue

            # 提取 modelname
            dir_name = os.path.basename(dirpath)
            try:
                _, model_name = dir_name.split("_", 1)
            except ValueError:
                print(f"跳过无效目录名：{dir_name}")
                continue

            # 找到数字最大的文件
            max_file = max(numbered_files, key=get_numeric_suffix)
            new_name = f"{model_name}.pth"
            old_path = os.path.join(dirpath, max_file)
            new_path = os.path.join(dirpath, new_name)

            # 重命名最大文件
            print(f"重命名: {old_path} -> {new_path}")
            os.rename(old_path, new_path)

            # 删除其他 .pth 文件
            for f in numbered_files:
                if f != max_file:
                    del_path = os.path.join(dirpath, f)
                    print(f"删除: {del_path}")
                    os.remove(del_path)

if __name__ == "__main__":
    base_folder = os.getcwd()  # 当前路径
    process_directory(base_folder)
