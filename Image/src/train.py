import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import pickle
from utils import SimpleCNN, seed_setting, create_dataset_directories,dataset_format_convert, load_dataset
from train_subset_model_for_verify import train_subset_model
from generate_nearest import calculate_distance
from Adv_Sn import train_Adv_Sn_model,get_adv_Du
from train_with_proof import train_with_proof
from train_with_forge import train_with_forge
from train_with_RL_du import train_random_label
from train_with_gradient_ascent import train_gradient_ascent
from train_with_fisher import fisher_with_alpha_selection
from train_with_fisher_hessian import fisher_hessian_with_alpha_selection
from certified_unlearn import certified_unlearn
import argparse

# tensorboard --logdir=../CIFAR10/logs --port=6006

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset selection for training')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                       choices=['SVHN', 'CIFAR10',  'Medical_32'],
                       help='name of selected dataset (default: CIFAR10)')
    args = parser.parse_args() # 解析命令行参数

    random_seed = 2568  # 设置种子
    seed_setting(random_seed) # 一键控制所有随机种子
    dataset_name = args.dataset_name # 全局使用的数据集
    global_lr=0.001 # 设置当前数据集全局学习率
    global_weight_decay=0.0001 # 设置当前数据集全局权重衰减率
    create_dataset_directories(dataset_name) # 为指定的数据集创建必要的目录结构

    train_transform, test_transform = dataset_format_convert(dataset_name) # 数据格式转换
    train_dataset, train_loader, test_dataset, test_loader = (  # 加载数据集以及data_loader
        load_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=8))

    # 将数据集按 4:1 划分
    total_len = len(train_dataset)
    dr_len = int(total_len * 0.8)
    du_len = total_len - dr_len
    Dr, Du = random_split(train_dataset, [dr_len, du_len])
    Dr_loader = DataLoader(Dr, batch_size=256, num_workers=8)
    Du_loader = DataLoader(Du, batch_size=256, num_workers=8)

    # 获取遗忘样本和保留样本索引
    retain_indices = Dr.indices
    unlearn_indices = Du.indices

    # 保存索引文件
    retain_path = f"../{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"../{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"

    if not os.path.exists(retain_path):
        # 保存当前划分的索引
        np.save(retain_path, np.array(retain_indices))
        np.save(unlearn_path, np.array(unlearn_indices))
        print(f"Saved indices to {retain_path} and {unlearn_path}")
    else:
        # 直接加载已有索引（可选）
        retain_indices = np.load(retain_path)
        unlearn_indices = np.load(unlearn_path)
        print(f"Loaded existing indices from {retain_path} and {unlearn_path}")

    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择gpu


    run_train_retrain_model = False  # 控制是否执行train_retrain_model
    run_train_pretrain_model = False  # 控制是否执行train_pretrain_model
    run_finetune_pretrain_model =False # 控制是否执行微调pretrain模型

    run_generate_nearest = False  # 控制是否执行generate_nearest
    run_train_Adv_Sn_model = False # 控制是否执行train_adv_model
    run_train_with_proof = False # 控制是否执行train_with_proof
    run_train_with_forge = False # 控制是否执行train_with_forge

    run_train_with_random_label = False  # 控制是否执行train_random_label
    run_train_with_random_label_finetune = False  # 控制是否执行train_random_label_finetune

    run_train_with_gradient_ascent = False   # 控制是否执行train_gradient_ascent
    run_train_with_gradient_ascent_finetune = False  # 控制是否执行train_gradient_ascent后续的微调操作

    run_train_with_fisher = False  # 控制是否执行train_train_fisher
    run_train_with_fisher_hessian = False  # 控制是否执行train_train_fisher_hessian

    run_certified_unlearn = False # certified_unlearn
    run_adv_retrain = False # adv_retrain

# 1.1执行train_retrain_model
    if run_train_retrain_model:

        Du_loader = DataLoader(Du, batch_size=256, shuffle=True, num_workers=8)
        Dr_loader = DataLoader(Dr, batch_size=256, shuffle=True, num_workers=8)
        # 为每个子集创建新的模型实例
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        # 默认使用相同的初始化模型
        init_model_path = f"../{dataset_name}/models/SimpleCNN_init_seed_{random_seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            # 如果文件不存在，先保存当前模型权重
            torch.save(model.state_dict(), init_model_path)
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))

        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 为每个子集配置单独的日志和保存路径
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_retrain/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_retrain/',
            'save_epoch': 50,
            'neg': False,
            'resume_epoch': 0,
        }
        # 自动创建目录
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        # 训练模型
        train_subset_model(
            model=model,
            train_loader=Dr_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=300,
            device=device_gpu,
            config=config
        )
        print("基于Dr训练的retrain模型训练完成!")

# 1.2执行train_pretrain_model
    if run_train_pretrain_model:

        # 为每个子集创建新的模型实例
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        # 默认使用和retrain相同的初始化模型
        init_model_path = f"../{dataset_name}/models/SimpleCNN_init_seed_{random_seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            # 如果文件不存在，先保存当前模型权重
            torch.save(model.state_dict(), init_model_path)
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))

        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 为每个子集配置单独的日志和保存路径
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_pretrain/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_pretrain/',
            'save_epoch': 50,
            'neg': False,  # 默认不反转梯度
            'resume_epoch': 0,
        }
        # 自动创建目录
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        # 训练模型
        train_subset_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,  
            num_epochs=300,
            device=device_gpu,
            config=config
        )
        print("pretrain模型训练完成!")


# 1.3对pretrain的模型基于Dr微调
    if run_finetune_pretrain_model:

        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)
        pretrained_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
        if os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"预训练模型 {pretrained_model_path} 不存在。请先运行预训练模型！")

        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 为每个子集配置单独的日志和保存路径
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_finetune_pretrain_model/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_finetune_pretrain_model/',
            'save_epoch': 50,
            'neg': False,  
            'resume_epoch': 0,  
        }
        # 自动创建目录
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        # 训练模型
        train_subset_model(
            model=model,
            train_loader=Dr_loader,
            test_loader=Du_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=300,
            device=device_gpu,
            config=config
        )
        print("pretrain_finetune模型训练完成!")


# 2.1 在此处计算相似度排序表
    if run_generate_nearest:
        print("开始计算样本相似度")
        # 计算样本相似度
        feat_dist = calculate_distance(dataset_name, train_dataset, device=device_gpu)

        # 保存相似度排序结果
        output_path = f'../{dataset_name}/data/{dataset_name}_train_datadist.pkl'
        with open(output_path, 'wb') as file:
            pickle.dump(feat_dist, file)
        print(f"Feature distance dictionary saved to {output_path}")

# 2.2重新训练Adv_Sn的一个模型
    if run_train_Adv_Sn_model:
        print("开始run_train_Adv_Sn_model")
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        # 默认使用相同的初始化模型
        init_model_path = f"../{dataset_name}/models/SimpleCNN_init_seed_{random_seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)

        # 配置超参数
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_adv_sn/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_adv_sn/',
            'save_epoch': 30,
            'neg': False,
            'resume_epoch': 0,
        }

        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        feat_dist_path = f'../{dataset_name}/data/{dataset_name}_train_datadist.pkl'  # 调用之前生成的相似度列表
        if os.path.exists(feat_dist_path):
            with open(feat_dist_path, 'rb') as file:
                feat_dist = pickle.load(file)
        else:
            raise ValueError(f"{feat_dist_path}没有找到，先执行run_generate_nearest!!!")

        train_Adv_Sn_model(
            model=model,
            train_dataset=train_dataset,
            Dr_loader=Dr_loader,
            Du_loader=Du_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            batch_size=256,  # 添加批次大小参数
            num_epochs=300,
            device=device_gpu,
            config=config,
            unlearn_indices=unlearn_indices,
            retain_indices=retain_indices,
            dist_dict=feat_dist,
        )

# 2.3 先train with proof
    if run_train_with_proof:

        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        # 默认使用相同的初始化模型
        init_model_path =  f"../{dataset_name}/models/SimpleCNN_init_seed_{random_seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)

        # 配置超参数
        config = {
            'save_optimizer_path': f'../{dataset_name}/optimizers/{dataset_name}_forge',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_forge',
            'save_batch_path': f'../{dataset_name}/batches/{dataset_name}_forge',
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_forge',
        }

        os.makedirs(config['save_optimizer_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)
        os.makedirs(config['save_batch_path'], exist_ok=True)
        os.makedirs(config['save_log_path'], exist_ok=True)

        #  开始训练
        train_with_proof(
            train_set=train_dataset,
            test_loader=test_loader,
            model=model,
            model_name='SimpleCNN',
            num_epochs=300,
            bs=2048,
            seed=random_seed,
            device=device_gpu,
            config=config,
            criterion=criterion,
            optimizer=optimizer,
        )
    
# 2.4 接着执行train_with_forge
    if run_train_with_forge:

        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)
        criterion = nn.CrossEntropyLoss()

        # 参数设置
        config = {
            'save_optimizer_path': f'../{dataset_name}/optimizers/{dataset_name}_forge',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_forge',
            'save_batch_path': f'../{dataset_name}/batches/{dataset_name}_forge',
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_forge',
        }

        os.makedirs(config['save_optimizer_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)
        os.makedirs(config['save_batch_path'], exist_ok=True)
        os.makedirs(config['save_log_path'], exist_ok=True)

        feat_dist_path = f'../{dataset_name}/data/{dataset_name}_train_datadist.pkl'  # 调用之前生成的相似度列表

        if os.path.exists(feat_dist_path):
            with open(feat_dist_path, 'rb') as file:
                feat_dist = pickle.load(file)
        else:
            raise ValueError(f"{feat_dist_path}没有找到，先执行run_generate_nearest!!!")

        train_with_forge(
            epoch = 300,
            unlearn_indices=unlearn_indices,
            Dr=Dr,
            train_dataset=train_dataset,
            feat_dist=feat_dist,
            config=config,
            process_id=0,  # 当前进程ID
            num_process=1,
            batch_size= 2048,
            dataset_name=dataset_name,
            lr=0.002,
            seed=random_seed,
            device=device_gpu,
            model=model,
            model_name='SimpleCNN',
            weight_decay=0.0002,
        )


# 3.1 执行train_random_label
    if run_train_with_random_label:
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        pretrained_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
        if os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"预训练模型 {pretrained_model_path} 不存在。请先运行预训练模型！")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)

        # 参数设置
        config = {
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_random_label/',
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_random_label/',
        }

        os.makedirs(config['save_model_path'], exist_ok=True)
        os.makedirs(config['save_log_path'], exist_ok=True)

        train_random_label(
            model=model,
            Du_dataset=Du,
            Du_loader=Du_loader,
            Dr_loader=Dr_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=300,
            device=device_gpu,
            num_classes=10,
            save_epoch=1,
            config=config,
        )
        print("基于Du随机标签的模型训练完成!")

# 3.2 执行train_random_label_fintune:
    if run_train_with_random_label_finetune:
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        init_model_path = f"../{dataset_name}/models/{dataset_name}_random_label/epoch_16.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 为每个子集配置单独的日志和保存路径
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_random_label_finetune/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_random_label_finetune/',
            'save_epoch': 50,
            'neg': False,  # 默认不反转梯度
            'resume_epoch': 0,  # 默认不进行断点重训
        }
        # 自动创建目录
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        # 训练模型
        train_subset_model(
            model=model,
            train_loader=Dr_loader,
            test_loader=Du_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,  # 学习率调度器，默认不使用
            num_epochs=300,
            device=device_gpu,
            config=config
        )
        print("random_label_fintune模型训练完成!")


# 4.1 执行train_gradient_ascent
    if run_train_with_gradient_ascent:
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        pretrained_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
        if os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"预训练模型 {pretrained_model_path} 不存在。请先运行预训练模型！")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.0000005)

        # 参数设置
        config = {
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_gradient_ascent/',
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_gradient_ascent/',
        }

        os.makedirs(config['save_model_path'], exist_ok=True)
        os.makedirs(config['save_log_path'], exist_ok=True)

        train_gradient_ascent(
            model=model,
            Du_dataset=Du,
            Du_loader=Du_loader,
            Dr_loader=Dr_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=50,
            device=device_gpu,
            num_classes=10,
            save_epoch=1,
            config=config,
        )
        print("基于Du梯度上升的模型训练完成!")

# 4.2 执行train_gradient_ascent_fintune
    if run_train_with_gradient_ascent_finetune:
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        init_model_path = f"../{dataset_name}/models/{dataset_name}_gradient_ascent/epoch_13.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 为每个子集配置单独的日志和保存路径
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_gradient_ascent_finetune/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_gradient_ascent_finetune/',
            'save_epoch': 50,
            'neg': False,
            'resume_epoch': 0,
        }
        # 自动创建目录
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        # 训练模型
        train_subset_model(
            model=model,
            train_loader=Dr_loader,
            test_loader=Du_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=300,
            device=device_gpu,
            config=config
        )
        print("gradient_ascent_finetune模型训练完成!")


# 5.1 fisher基础版本
    if run_train_with_fisher:

            model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)
            # 默认使用相同的初始化模型
            init_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
            if os.path.exists(init_model_path):
                model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
            else:
                raise FileNotFoundError(
                    f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

            # 参数设置
            config = {
                'save_model_path': f'../{dataset_name}/models/{dataset_name}_fisher/',
            }

            os.makedirs(config['save_model_path'], exist_ok=True)

            best_alpha, best_accuracy = fisher_with_alpha_selection(
                model=model,
                retain_loader=Dr_loader,
                Du_loader=Du_loader,  # 用于测试遗忘效果
                Dr_loader=Dr_loader,  # 用于测试保留效果
                test_loader=test_loader,  # 用于测试泛化能力
                device=device_gpu,
                config=config,
            )

# 5.2 fisher_hessian方法
    if run_train_with_fisher_hessian:
            criterion = nn.CrossEntropyLoss()
            model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)
            # 加载预训练模型
            init_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
            if os.path.exists(init_model_path):
                model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
            else:
                raise FileNotFoundError(
                    f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

            # 参数设置
            config = {
                'save_model_path': f'../{dataset_name}/models/{dataset_name}_fisher_hessian/',
            }
            os.makedirs(config['save_model_path'], exist_ok=True)

            best_alpha, best_accuracy = fisher_hessian_with_alpha_selection(
                model=model,
                retain_loader=Dr_loader,
                Du_loader=Du_loader,  # 用于测试遗忘效果
                Dr_loader=Dr_loader,  # 用于测试保留效果
                test_loader=test_loader,  # 用于测试泛化能力
                criterion=criterion,
                device=device_gpu,
                config=config,
            )
            print("所有alpha候选值的fisher_hessian训练完成!")


# 6.1 certified方法
    if run_certified_unlearn:

        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)
        # 默认使用相同的初始化模型
        init_model_path = f"../{dataset_name}/models/{dataset_name}_pretrain/epoch_300.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        config = {
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_certified_unlearn/'
        }

        os.makedirs(config['save_model_path'], exist_ok=True)

        certified_unlearn(
            Dr_loader=Dr_loader,
            Dr=Dr,
            model=model,
            device=device_gpu,
            path=config['save_model_path']+"certified_ul.pth"
        )

# 6.2 adv_retrain方法
    if run_adv_retrain:

        print("开始run_adv_retrain")
        model = SimpleCNN(dataset_name=dataset_name, num_classes=10).to(device_gpu)

        # 默认使用相同的初始化模型
        init_model_path =  f"../{dataset_name}/models/SimpleCNN_init_seed_{random_seed}.pth"
        if os.path.exists(init_model_path):
            model.load_state_dict(torch.load(init_model_path, map_location='cuda:0'))
        else:
            raise FileNotFoundError(
                f"Model file {init_model_path} not found. Please ensure the model file exists before continuing.")

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=global_lr, weight_decay=global_weight_decay)

        # 配置超参数
        config = {
            'save_log_path': f'../{dataset_name}/logs/{dataset_name}_adv_retrain/',
            'save_model_path': f'../{dataset_name}/models/{dataset_name}_adv_retrain/',
            'save_epoch': 50,
            'neg': False,
            'resume_epoch': 0,
        }

        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        feat_dist_path = f'../{dataset_name}/data/{dataset_name}_train_datadist.pkl'  # 调用之前生成的相似度列表
        if os.path.exists(feat_dist_path):
            with open(feat_dist_path, 'rb') as file:
                feat_dist = pickle.load(file)
        else:
            raise ValueError(f"{feat_dist_path}没有找到，先执行run_generate_nearest!!!")

        Du_r = get_adv_Du(Dr, dist_dict=feat_dist, unlearn_indices=unlearn_indices, retrain_indices=retain_indices)
        Dr_indices = set(Dr.indices)
        Du_r_global_indices = set(Dr.indices[i] for i in Du_r.indices)
        Du_indices = set(Du.indices)

        # 拼接新的训练集
        combined_indices = list(Dr_indices) + list(Du_r_global_indices)
        train_new = torch.utils.data.Subset(train_dataset, combined_indices)

        # 从 Dr 中去掉 Du_r
        Dr_new_indices = Dr_indices - Du_r_global_indices
        # 再加上 Du
        Dr_new_indices |= Du_indices

        # 转成 list 并随机打乱
        Dr_new_indices = list(Dr_new_indices)
        random.shuffle(Dr_new_indices)

        # 构造新的训练集
        Dr_new = torch.utils.data.Subset(train_dataset, Dr_new_indices)
        Dr_new_loader = DataLoader(Dr_new, batch_size=256, num_workers=8)
        train_subset_model(
            model=model,
            train_loader=Dr_new_loader,
            test_loader=Du_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=300,
            device=device_gpu,
            config=config
        )

# tensorboard --logdir=../CIFAR10/logs --port=6006



























