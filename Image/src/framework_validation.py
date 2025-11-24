# -*- coding: utf-8 -*-
"""
Compute agreement between each mix50 model and all reference models (SIM set).
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
import os
import sys
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from utils import SimpleCNN, seed_setting,dataset_format_convert, load_dataset
from evalution_metrics import evaluate_with_prob_average_return_data
import pandas as pd
import argparse
import torch.nn.functional as F

def ensemble_predictions_prob_average(attack_models, inputs, device):
    """
    使用 softmax 概率的平均来综合多个攻击模型的预测
    :param attack_models: 攻击模型的列表
    :param inputs: 输入数据
    :param device: 使用的计算设备
    :return: 综合后的概率分布（经过 softmax 平均）
    """
    # 获取每个攻击模型的 softmax 概率输出
    probs_ensemble = torch.stack(
        [F.softmax(model(inputs.to(device)), dim=1) for model in attack_models],
        dim=0
    ).mean(dim=0)  # 对概率分布求平均

    return probs_ensemble


def compute_agreement_strict(target_model, attack_models, dataloader, device):
    """
    计算 target_model 与 attack_models(ensemble) 的agreement。
    使用 evaluate_with_prob_average_all 的原始逻辑。
    """
    target_model.eval()
    for m in attack_models:
        m.eval()

    total_agree = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            logits_target = target_model(inputs)
            probs_ensemble = ensemble_predictions_prob_average(attack_models, inputs, device)
            batch_agreement = (torch.argmax(logits_target, dim=1) == torch.argmax(probs_ensemble, dim=1)).sum().item()

            total_agree += batch_agreement
            total_samples += inputs.size(0)

    return total_agree / total_samples if total_samples > 0 else 0.0


def compute_agreement_vs_collection(add_exp_name,mix_models, mix_names, ref_models, ref_names, dataloader, device, threshold=0.9, save_dir="./results"):
    """计算每个mix模型与所有参考模型的agreement"""
    results_matrix = np.zeros((len(mix_models), len(ref_models)), dtype=float)
    bool_matrix = np.zeros((len(mix_models), len(ref_models)), dtype=bool)

    print(f"🔍 Computing agreement between {len(mix_models)} mix models and {len(ref_models)} reference models...")

    for i, mix_model in enumerate(mix_models):
        for j, ref_model in enumerate(ref_models):
            agr = compute_agreement_strict(mix_model, [ref_model], dataloader, device)
            results_matrix[i, j] = agr
            bool_matrix[i, j] = agr >= threshold
            print(f"[{mix_names[i]} vs {ref_names[j]}] Agreement={agr:.4f} → {bool_matrix[i,j]}")

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"{add_exp_name}_vs_SIM_agreement_thresh_{threshold}.npz"),
             agreement_matrix=results_matrix,
             bool_matrix=bool_matrix,
             mix_names=mix_names,
             ref_names=ref_names)

    df_val = pd.DataFrame(results_matrix, index=mix_names, columns=ref_names)
    df_bool = pd.DataFrame(bool_matrix, index=mix_names, columns=ref_names)
    df_val.to_csv(os.path.join(save_dir, f"{add_exp_name}_vs_SIM_agreement_values_{threshold}.csv"),float_format='%.5f')
    df_bool.to_csv(os.path.join(save_dir, f"{add_exp_name}_vs_SIM_agreement_bool_{threshold}.csv"))

    print(f"✅ Results saved to {save_dir}")
    return results_matrix, bool_matrix


if __name__ == "__main__":
    # ==============================
    # 配置
    # ==============================

    parser = argparse.ArgumentParser(description='Ablation Experiment')
    parser.add_argument('--dataset', type=str, default='Medical_32',
                       choices=['SVHN', 'CIFAR10',  'Medical_32'])
    parser.add_argument('--add_exp_num', type=int, default=3,
                        choices=[1, 2, 3, 4],
                       help='1:dr50 du50, 2:dr 80')
    args = parser.parse_args()

    dataset_name = args.dataset
    add_exp_num = args.add_exp_num

    if add_exp_num == 1:
        add_exp_name = 'Mixed_Dr50_Du50'
    elif add_exp_num == 2:
        add_exp_name = 'DrOnly_80'
    elif add_exp_num == 3:
        add_exp_name = 'Mixed_Dr40_Du90'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = 0.9001

    # if dataset_name == 'Medical_32':
    #     threshold = 0.9167
    # elif dataset_name == 'SVHN':
    #     threshold = 0.8816
    # elif dataset_name == 'CIFAR10':
    #     threshold = 0.6567

    # 11.10修正参数，适用于40Dr+90Du
    if dataset_name == 'Medical_32':
        threshold = 0.9167  # 沿用agreement rate 即可实现效果
    elif dataset_name == 'SVHN':
        threshold = 0.8985 # 调高阈值
    elif dataset_name == 'CIFAR10':
        threshold = 0.6750 # 调高阈值

    num_classes = 2 if dataset_name == 'Medical_32' else 10
    random_seed = 2568  # 设置种子

    # ==============================
    # 数据集加载
    # ==============================
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择gpu

    train_transform, test_transform = dataset_format_convert(dataset_name)  # 数据格式转换
    train_dataset, train_loader, test_dataset, test_loader = (  # 加载数据集以及data_loader
        load_dataset(dataset_name,train_transform,test_transform,batch_size=256,num_workers=8))

    retain_path = f"../{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"../{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"

    if os.path.exists(retain_path) and os.path.exists(unlearn_path):
        retain_indices = np.load(retain_path)
        unlearn_indices = np.load(unlearn_path)
        Dr = Subset(train_dataset, retain_indices)
        Du = Subset(train_dataset, unlearn_indices)
        print("Loaded saved splits")
    else:
        raise FileNotFoundError("Split indices not found. Run main_cifar.py first!")
    # 评估模型
    test_dr = DataLoader(Dr, batch_size=256, shuffle=False)
    test_du = DataLoader(Du, batch_size=256, shuffle=False)


    # ==============================
    # 加载 reference 模型集合（SIM）
    # ==============================
    x_values = [5, 10, 20, 30, 40, 50, 60]  # 定义x的值
    ref_models, ref_names = [], []

    for x in x_values:
        path = f"../{dataset_name}/models/{dataset_name}_SIM_Ablation_{x}/Dr1/epoch_300.pth"
        if os.path.exists(path):
            m = SimpleCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            ref_models.append(m)
            ref_names.append(f"SIM_Ablation_{x}")
            print(f"  Loaded: SIM_Ablation_{x}")
        else:
            print(f"  Warning: {path} not found")

    print(f"✅ Loaded {len(ref_models)} reference models")


    # ==============================
    # 加载 50 个 mix 模型
    # ==============================
    mix_models, mix_names = [], []

    for sd in range(10086, 10136):  # 10086到10135，共50个
        # path = f"../{dataset_name}/models/{dataset_name}_{add_exp_name}/dynamic_seed_{sd}/epoch_300.pth"
        path = f"../{dataset_name}/models/{dataset_name}_{add_exp_name}/dynamic_seed_{sd}/epoch_300.pth"
        if os.path.exists(path):
            m = SimpleCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            mix_models.append(m)
            mix_names.append(f"dynamic_seed_{sd}")
            print(f"  Loaded: dynamic_seed_{sd}")
        else:
            print(f"  Warning: {path} not found")

    print(f"✅ Loaded {len(mix_models)} mix50 models")

    # ==============================
    # 计算 agreement 并保存
    # ==============================
    results_dir = f"../{dataset_name}/verification_results/add_exp_verification_results_du/"
    compute_agreement_vs_collection(
        add_exp_name=add_exp_name,
        mix_models=mix_models,
        mix_names=mix_names,
        ref_models=ref_models,
        ref_names=ref_names,
        dataloader=test_du,
        device=device,
        threshold=threshold,
        save_dir=results_dir
    )
