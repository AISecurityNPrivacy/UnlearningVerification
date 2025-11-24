
## 执行指南

### 基本执行步骤：

1. **环境准备**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision numpy tqdm scikit-learn tensorboard pillow
```

2. **运行基础实验**
```bash
# 使用默认设置（CIFAR10数据集）
python train_f.py

# 指定其他数据集
python train_f.py --dataset_name SVHN
python train_f.py --dataset_name Medical_32
```

3. **自定义实验配置**

编辑 `train_f.py` 第71-90行的控制变量：
```python
# 选择要运行的实验
run_train_retrain_model = True  # 重训练基准
run_train_with_random_label = True  # 随机标签遗忘
run_train_with_fisher = True  # Fisher信息遗忘
# ... 更多选项
```

4. **监控训练进度**
```bash
# 启动TensorBoard
tensorboard --logdir=../[dataset_name]/logs --port=6006

# 在浏览器中查看
http://localhost:6006
```

### 批量实验脚本示例：

```bash
#!/bin/bash
# run_experiments.sh

# 定义数据集列表
datasets=("CIFAR10" "SVHN")

# 运行所有数据集的实验
for dataset in "${datasets[@]}"
do
    echo "Running experiments on $dataset..."
    python train_f.py --dataset_name $dataset
done
```
