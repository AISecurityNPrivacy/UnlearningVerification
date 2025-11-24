import os
import argparse
import random
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn, optim

from utils import *
from train_scripts.train_subset_model_for_verify import train_subset_model
from train_scripts.generate_nearest import calculate_distance_text
from train_scripts.Adv_Sn import get_adv_Dv, get_adv_Du, train_Adv_Sn_model
from train_scripts.train_with_proof import train_with_proof
from train_scripts.train_with_forge import train_with_forge
from train_scripts.train_with_random_label import train_random_label
from train_scripts.train_with_gradient_ascent_2 import train_gradient_ascent_2
from train_scripts.train_with_fisher import fisher
from train_scripts.train_with_fisher_hessian import fisher_hessian
from train_scripts.certified_unlearn import certified_unlearn

from collections import defaultdict

def split_unbalance_dataset(train_dataset, du_rate=0.2):
    total_len = len(train_dataset)
    du_len = int(total_len * du_rate)

    # 获取所有 label
    labels = [train_dataset[i][1] for i in range(total_len)]

    # 按标签分类索引
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_value = label.item() if isinstance(label, torch.Tensor) else label
        label_indices[label_value].append(idx)

    # Du 中 label=0 数量
    du_label0_num = du_len // 2
    assert len(label_indices[0]) >= du_label0_num, "标签0样本不足构建 unbalance Du"

    # 随机抽样 label=0
    du_label0_indices = torch.randperm(len(label_indices[0]))[:du_label0_num].tolist()
    du_label0_indices = [label_indices[0][i] for i in du_label0_indices]

    # 其他标签索引
    other_indices = []
    for l in label_indices:
        if l != 0:
            other_indices.extend(label_indices[l])

    du_other_num = du_len - du_label0_num
    assert len(other_indices) >= du_other_num, "其他标签数量不足构建 unbalance Du"

    du_other_indices = torch.randperm(len(other_indices))[:du_other_num].tolist()
    du_other_indices = [other_indices[i] for i in du_other_indices]

    # 最终 Du
    du_indices = du_label0_indices + du_other_indices

    # Dr = 剩余
    dr_indices = list(set(range(total_len)) - set(du_indices))

    return dr_indices, du_indices

def init_model(dataset_name, num_classes, scene, device):
    if scene == "SCNN":
        return TextSCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
    elif scene == "RCNN":
        return TextRCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
    else:
        return TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)


def get_folder_name(scene, du_rate):
    return str(du_rate) if scene == "basic" else scene


def get_roots(scene, du_rate, dataset):
    folder = get_folder_name(scene, du_rate)
    base_dir = os.path.join("models", str(folder), dataset)
    model_root = os.path.join(base_dir, "model")
    data_root = os.path.join(base_dir, "data")
    logs_root = os.path.join(base_dir, "logs")
    optim_root = os.path.join(base_dir, "optimizers")
    batch_root = os.path.join(base_dir, "batches")
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(optim_root, exist_ok=True)
    os.makedirs(batch_root, exist_ok=True)
    return model_root, data_root, logs_root, optim_root, batch_root


def model_dir(model_root, dataset, method):
    d = os.path.join(model_root, f"{dataset}_{method}")
    os.makedirs(d, exist_ok=True)
    return d


def logs_dir(logs_root, dataset, method):
    d = os.path.join(logs_root, f"{dataset}_{method}")
    os.makedirs(d, exist_ok=True)
    return d


def pick_latest_checkpoint(save_dir):
    if not os.path.isdir(save_dir):
        return None
    files = [f for f in os.listdir(save_dir) if f.endswith(".pth")]
    if not files:
        return None

    # priority: best.pth > epoch_X.pth (max X) > numeric.pth (max) > others
    if "best.pth" in files:
        return "best.pth"

    epoch_files = []
    num_files = []
    others = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.startswith("epoch_"):
            suffix = name[6:]
            if suffix.isdigit():
                epoch_files.append((int(suffix), f))
            else:
                others.append(f)
        elif name.isdigit():
            num_files.append((int(name), f))
        else:
            others.append(f)

    if epoch_files:
        return max(epoch_files)[1]
    if num_files:
        return max(num_files)[1]
    return others[0] if others else None


def copy_as_method(save_dir, method_name):
    ckpt = pick_latest_checkpoint(save_dir)
    if ckpt is None:
        print(f"[WARN] no checkpoint found in {save_dir}, cannot create {method_name}.pth")
        return
    src = os.path.join(save_dir, ckpt)
    dst = os.path.join(save_dir, f"{method_name}.pth")
    if src != dst:
        import shutil
        shutil.copy2(src, dst)
        print(f"[INFO] copied {src} -> {dst}")
    else:
        print(f"[INFO] checkpoint already named {method_name}.pth in {save_dir}")



def create_dataset_directories(dataset_name):
    for d in ["models", "logs", "data"]:
        os.makedirs(d, exist_ok=True)


def generate_single_verify_subsets(Dr_dataset, Dr_indices_range, subset_len, replace=False):
    # 从 Dr 的范围选 index
    if replace:
        chosen = np.random.choice(Dr_indices_range, subset_len, replace=True)
    else:
        chosen = np.random.choice(Dr_indices_range, subset_len, replace=False)

    # Dr_dataset.indices = Dr 对应的 “全局索引”
    chosen_global = [Dr_dataset.indices[i] for i in chosen]

    return torch.utils.data.Subset(Dr_dataset.dataset, chosen_global)


def generate_single_subset_dataloader(subset, batch_size=256, shuffle=True, num_workers=0):
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--dataset", type=str, default="BBCNews",
                        choices=["BBCNews", "IMDb", "AGNews"])
    parser.add_argument('-scene', "--scenario", type=str, default="basic",
                        choices=["basic", "SCNN", "RCNN", "unbalance"])
    parser.add_argument('-du_r', "--du_rate", type=float, default=0.2,
                        choices=[0.1, 0.2, 0.3],
                        help="Du 占总训练集比例，仅 basic 场景生效")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    dataset = args.dataset
    scene = args.scenario
    du_rate = args.du_rate
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    random_seed = 2568
    seed_setting(random_seed)

    num_classes = 2 if dataset == "IMDb" else 5

    # 目录
    model_root, data_root, logs_root, optim_root, batch_root = get_roots(scene, du_rate, dataset)

    # 数据集
    create_dataset_directories(dataset)
    train_dataset, train_loader, test_dataset, test_loader = load_dataset(dataset, batch_size=256, num_workers=0)
    if scene == 'unbalance':
        dr_indices, du_indices = split_unbalance_dataset(train_dataset, du_rate=0.2)
        Dr = torch.utils.data.Subset(train_dataset, dr_indices)
        Du = torch.utils.data.Subset(train_dataset, du_indices)
    else:
        total_len = len(train_dataset)
        effective_du_rate = du_rate if scene == "basic" else 0.2
        dr_len = int(total_len * (1.0 - effective_du_rate))
        du_len = total_len - dr_len
        Dr, Du = random_split(train_dataset, [dr_len, du_len])
    Dr_loader = DataLoader(Dr, batch_size=256, shuffle=True, num_workers=0)
    Du_loader = DataLoader(Du, batch_size=256, shuffle=True, num_workers=0)

    # 再划分 Dr1 / Dr2 (agree base)
    dr_total_len = len(Dr)
    dr1_len = dr_total_len // 2
    dr2_len = dr_total_len - dr1_len
    Dr1, Dr2 = random_split(Dr, [dr1_len, dr2_len])
    Dr1_loader = DataLoader(Dr1, batch_size=256, shuffle=True, num_workers=0)
    Dr2_loader = DataLoader(Dr2, batch_size=256, shuffle=True, num_workers=0)

    retain_indices = np.array(Dr.indices)
    unlearn_indices = np.array(Du.indices)

    retain_path = os.path.join(data_root, f"retain_splits_seed_{random_seed}.npy")
    unlearn_path = os.path.join(data_root, f"unlearn_splits_seed_{random_seed}.npy")
    np.save(retain_path, retain_indices)
    np.save(unlearn_path, unlearn_indices)
    print(f"[INFO] Saved retain/unlearn indices to {retain_path} / {unlearn_path}")

    # 计算相似度（如已存在则直接加载）
    dist_path = os.path.join(data_root, f"{dataset}_train_datadist.pkl")
    if os.path.exists(dist_path):
        with open(dist_path, "rb") as f:
            feat_dist = pickle.load(f)
        print(f"[INFO] Loaded distance dict from {dist_path}")
    else:
        print("[INFO] Calculating distance dict ...")
        feat_dist = calculate_distance_text(dataset, train_dataset, device=device)
        with open(dist_path, "wb") as f:
            pickle.dump(feat_dist, f)
        print(f"[INFO] Saved distance dict to {dist_path}")

    # ------------------------
    #  1. Base agree1 / agree2
    # ------------------------
    print("\n=== Training agree1 / agree2 ===")
    for name, loader in [("agree1", Dr1_loader), ("agree2", Dr2_loader)]:
        print(f"[INFO] Training {name} ...")
        model = init_model(dataset, num_classes, scene, device)
        save_model_path = model_dir(model_root, dataset, name)
        save_log_path = logs_dir(logs_root, dataset, name)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        config = {
            "save_log_path": save_log_path,
            "save_model_path": save_model_path,
            "save_epoch": 30,
            "neg": False,
            "resume_epoch": 0,
        }
        train_subset_model(
            model=model,
            train_loader=loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=90,
            device=device,
            config=config,
        )
        copy_as_method(save_model_path, name)

    # ------------------------
    #  2. pretrain
    # ------------------------
    print("\n=== Training pretrain ===")
    model = init_model(dataset, num_classes, scene, device)
    pretrain_model_path = model_dir(model_root, dataset, "pretrain")
    pretrain_log_path = logs_dir(logs_root, dataset, "pretrain")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "save_log_path": pretrain_log_path,
        "save_model_path": pretrain_model_path,
        "save_epoch": 10,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=100,
        device=device,
        config=config,
    )
    copy_as_method(pretrain_model_path, "pretrain")

    # ------------------------
    #  3. pretrain_finetune (在 Dr 上微调)
    # ------------------------
    print("\n=== Training pretrain_finetune ===")
    model = init_model(dataset, num_classes, scene, device)
    # 从 pretrain 目录中取最后一个 ckpt
    pre_ckpt = pick_latest_checkpoint(pretrain_model_path)
    if pre_ckpt is None:
        raise RuntimeError(f"No checkpoint found in {pretrain_model_path} for pretrain_finetune")
    model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    pt_ft_model_path = model_dir(model_root, dataset, "pretrain_finetune")
    pt_ft_log_path = logs_dir(logs_root, dataset, "pretrain_finetune")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "save_log_path": pt_ft_log_path,
        "save_model_path": pt_ft_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=Dr_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=150,
        device=device,
        config=config,
    )
    copy_as_method(pt_ft_model_path, "pretrain_finetune")

    # ------------------------
    #  4. SIM_0.05 ~ SIM_0.60
    # ------------------------
    print("\n=== Training SIM_xx models ===")
    sim_root = os.path.join(model_root, f"{dataset}_SIM")
    os.makedirs(sim_root, exist_ok=True)
    Dr_indices_range = list(range(dr_len))
    for ratio in np.arange(0.05, 0.65, 0.05):
        ratio = round(float(ratio), 2)
        print(f"[INFO] Training SIM_{ratio:.2f}")
        subset_len = int(dr_len * ratio)
        subset = generate_single_verify_subsets(Dr, Dr_indices_range, subset_len, replace=False)
        subset_loader = generate_single_subset_dataloader(subset=subset, batch_size=256, shuffle=True, num_workers=0)

        model = init_model(dataset, num_classes, scene, device)
        sim_dir = os.path.join(sim_root, f"SIM_{ratio:.2f}")
        os.makedirs(sim_dir, exist_ok=True)
        sim_log_dir = logs_dir(logs_root, dataset, f"SIM_{ratio:.2f}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        config = {
            "save_log_path": sim_log_dir,
            "save_model_path": sim_dir,
            "save_epoch": 30,
            "neg": False,
            "resume_epoch": 0,
        }
        train_subset_model(
            model=model,
            train_loader=subset_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=100,
            device=device,
            config=config,
        )
        # 不需要特定文件名，verify 里是遍历子目录取第一个 .pth

    # ------------------------
    #  5. Adv_SIM (Mv_adv)
    # ------------------------
    print("\n=== Training Adv_SIM ===")
    model = init_model(dataset, num_classes, scene, device)
    Dv = get_adv_Dv(Dr, feat_dist, retrain_indices=retain_indices, unlearn_indices=unlearn_indices)
    Dv_loader = generate_single_subset_dataloader(subset=Dv, batch_size=256, shuffle=True, num_workers=0)
    adv_sim_model_path = model_dir(model_root, dataset, "Adv_SIM")
    adv_sim_log_path = logs_dir(logs_root, dataset, "Adv_SIM")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "save_log_path": adv_sim_log_path,
        "save_model_path": adv_sim_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=Dv_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=100,
        device=device,
        config=config,
    )
    copy_as_method(adv_sim_model_path, "Adv_SIM")

    # ------------------------
    #  6. retrain (基于 Dr)
    # ------------------------
    print("\n=== Training retrain ===")
    model = init_model(dataset, num_classes, scene, device)
    retrain_model_path = model_dir(model_root, dataset, "retrain")
    retrain_log_path = logs_dir(logs_root, dataset, "retrain")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "save_log_path": retrain_log_path,
        "save_model_path": retrain_model_path,
        "save_epoch": 25,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=Dr_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=100,
        device=device,
        config=config,
    )
    copy_as_method(retrain_model_path, "retrain")

    # ------------------------
    #  7. proof (train_with_proof) + forge (train_with_forge)
    # ------------------------
    print("\n=== Training proof (for forge) ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain for stability
    pre_ckpt = pick_latest_checkpoint(pretrain_model_path)
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    forge_opt_path = os.path.join(optim_root, f"{dataset}_forge")
    forge_model_path = model_dir(model_root, dataset, "forge")
    forge_batch_path = os.path.join(batch_root, f"{dataset}_forge")
    forge_log_path = logs_dir(logs_root, dataset, "forge")
    os.makedirs(forge_opt_path, exist_ok=True)
    os.makedirs(forge_batch_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    proof_config = {
        "save_optimizer_path": forge_opt_path,
        "save_model_path": forge_model_path,
        "save_batch_path": forge_batch_path,
        "save_log_path": forge_log_path,
    }

    train_with_proof(
        train_set=train_dataset,
        test_loader=test_loader,
        model=model,
        model_name="TextCNN",
        num_epochs=50,
        bs=256,
        seed=random_seed,
        device=device,
        config=proof_config,
        criterion=criterion,
        optimizer=optimizer,
    )

    print("\n=== Training forge ===")
    # forge 使用 feat_dist + proof 过程中保存的 batch/optimizer 等
    forge_config = proof_config
    train_with_forge(
        epoch=50,
        unlearn_indices=unlearn_indices,
        Dr=Dr,
        train_dataset=train_dataset,
        feat_dist=feat_dist,
        config=forge_config,
        process_id=0,
        num_process=1,
        batch_size=256,
        dataset_name=dataset,
        lr=0.002,
        seed=random_seed,
        device=device,
        model=model,
        model_name="TextCNN",
        weight_decay=0.0002,
    )
    copy_as_method(forge_model_path, "forge")

    # ------------------------
    #  8. relabel (random_label) + relabel_finetune
    # ------------------------
    print("\n=== Training relabel (random_label) ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain
    pre_ckpt = pick_latest_checkpoint(pretrain_model_path)
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    relabel_model_path = model_dir(model_root, dataset, "relabel")
    relabel_log_path = logs_dir(logs_root, dataset, "relabel")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    relabel_config = {
        "save_model_path": relabel_model_path,
        "save_log_path": relabel_log_path,
    }
    train_random_label(
        model=model,
        unlearn_dataset=Du,
        Du_loader=Du_loader,
        Dr_loader=Dr_loader,
        batch_size=256,
        unlearn_indices=unlearn_indices,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device,
        resume_epoch=0,
        num_classes=num_classes,
        save_epoch=1,
        config=relabel_config,
    )
    # 选一个 ckpt 作为 relabel.pth
    copy_as_method(relabel_model_path, "relabel")

    print("\n=== Training relabel_finetune ===")
    model = init_model(dataset, num_classes, scene, device)
    # 用 early epoch（这里取 5）初始化
    relabel_init = os.path.join(relabel_model_path, "5.pth")
    if os.path.exists(relabel_init):
        model.load_state_dict(torch.load(relabel_init, map_location=device))
    else:
        ckpt = pick_latest_checkpoint(relabel_model_path)
        if ckpt is None:
            raise RuntimeError(f"No checkpoint for relabel_finetune in {relabel_model_path}")
        model.load_state_dict(torch.load(os.path.join(relabel_model_path, ckpt), map_location=device))

    relabel_ft_model_path = model_dir(model_root, dataset, "relabel_finetune")
    relabel_ft_log_path = logs_dir(logs_root, dataset, "relabel_finetune")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    relabel_ft_config = {
        "save_log_path": relabel_ft_log_path,
        "save_model_path": relabel_ft_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=Dr_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=150,
        device=device,
        config=relabel_ft_config,
    )
    copy_as_method(relabel_ft_model_path, "relabel_finetune")

    # ------------------------
    #  9. gradient_ascent (用 gradient_ascent_2 实现) + gradient_ascent_finetune
    # ------------------------
    print("\n=== Training gradient_ascent (via gradient_ascent_2) ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain
    pre_ckpt = pick_latest_checkpoint(pretrain_model_path)
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    ga_model_path = model_dir(model_root, dataset, "gradient_ascent")
    ga_log_path = logs_dir(logs_root, dataset, "gradient_ascent")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ga_config = {
        "save_model_path": ga_model_path,
        "save_log_path": ga_log_path,
    }
    train_gradient_ascent_2(
        model=model,
        unlearn_dataset=Du,
        Du_loader=test_loader,
        Dr_loader=Dr_loader,
        batch_size=256,
        unlearn_indices=unlearn_indices,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device,
        resume_epoch=0,
        num_classes=num_classes,
        save_epoch=1,
        config=ga_config,
    )
    copy_as_method(ga_model_path, "gradient_ascent")

    print("\n=== Training gradient_ascent_finetune ===")
    model = init_model(dataset, num_classes, scene, device)
    ga_init = os.path.join(ga_model_path, "5.pth")
    if os.path.exists(ga_init):
        model.load_state_dict(torch.load(ga_init, map_location=device))
    else:
        ckpt = pick_latest_checkpoint(ga_model_path)
        if ckpt is None:
            raise RuntimeError(f"No checkpoint for gradient_ascent_finetune in {ga_model_path}")
        model.load_state_dict(torch.load(os.path.join(ga_model_path, ckpt), map_location=device))

    ga_ft_model_path = model_dir(model_root, dataset, "gradient_ascent_finetune")
    ga_ft_log_path = logs_dir(logs_root, dataset, "gradient_ascent_finetune")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    ga_ft_config = {
        "save_log_path": ga_ft_log_path,
        "save_model_path": ga_ft_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }
    train_subset_model(
        model=model,
        train_loader=Dr_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=150,
        device=device,
        config=ga_ft_config,
    )
    copy_as_method(ga_ft_model_path, "gradient_ascent_finetune")

    # ------------------------
    # 10. fisher / fisher_hessian
    # ------------------------
    print("\n=== Training fisher ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain (一般 fisher 类方法需要预训练模型)
    pre_ckpt = pick_latest_checkpoint(pretrain_model_path)
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    fisher_model_path = model_dir(model_root, dataset, "fisher")
    fisher_config = {
        "save_model_path": fisher_model_path,
    }
    fisher(
        model=model,
        retain_loader=Dr_loader,
        device=device,
        alpha=0.000005,
        config=fisher_config,
    )
    copy_as_method(fisher_model_path, "fisher")

    print("\n=== Training fisher_hessian ===")
    model = init_model(dataset, num_classes, scene, device)
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    fisher_hessian_model_path = model_dir(model_root, dataset, "fisher_hessian")
    fisher_hessian_config = {
        "save_model_path": fisher_hessian_model_path,
    }
    criterion = nn.CrossEntropyLoss()
    fisher_hessian(
        model=model,
        retain_loader=Dr_loader,
        device=device,
        criterion=criterion,
        alpha=0.000009,
        config=fisher_hessian_config,
    )
    copy_as_method(fisher_hessian_model_path, "fisher_hessian")

    # ------------------------
    # 11. certified_unlearn
    # ------------------------
    print("\n=== Training certified_unlearn ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    cert_model_path = model_dir(model_root, dataset, "certified_unlearn")
    os.makedirs(cert_model_path, exist_ok=True)
    cert_path = os.path.join(cert_model_path, "certified_unlearn.pth")
    certified_unlearn(
        Dr_loader=Dr_loader,
        Dr=Dr,
        model=model,
        device=device,
        path=cert_path,
    )

    # ------------------------
    # 12. attack_retrain (Adv_Sn)
    # ------------------------
    print("\n=== Training attack_retrain (Adv_Sn) ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    attack_model_path = model_dir(model_root, dataset, "attack_retrain")
    attack_log_path = logs_dir(logs_root, dataset, "attack_retrain")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    attack_config = {
        "save_log_path": attack_log_path,
        "save_model_path": attack_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }

    train_Adv_Sn_model(
        model=model,
        train_dataset=train_dataset,
        Dr_loader=Dr_loader,
        Du_loader=Du_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        batch_size=256,
        num_epochs=150,
        device=device,
        config=attack_config,
        unlearn_indices=unlearn_indices,
        retain_indices=retain_indices,
        dist_dict=feat_dist,
    )
    copy_as_method(attack_model_path, "attack_retrain")

    # ------------------------
    # 13. adv_retrain
    # ------------------------
    print("\n=== Training adv_retrain ===")
    model = init_model(dataset, num_classes, scene, device)
    # init from pretrain
    if pre_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(pretrain_model_path, pre_ckpt), map_location=device))

    adv_retrain_model_path = model_dir(model_root, dataset, "adv_retrain")
    adv_retrain_log_path = logs_dir(logs_root, dataset, "adv_retrain")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 使用 get_adv_Du 构建 Du_r
    Du_r = get_adv_Du(Dr, dist_dict=feat_dist, unlearn_indices=unlearn_indices, retrain_indices=retain_indices)

    # 转成全局索引
    Dr_indices_set = set(retain_indices.tolist())
    Du_r_global_indices = set(retain_indices[i] for i in Du_r.indices)
    Du_indices_set = set(unlearn_indices.tolist())

    # 新训练集：Dr + Du_r
    combined_indices = list(Dr_indices_set) + list(Du_r_global_indices)
    train_new = Subset(train_dataset, combined_indices)

    # Dr_new = (Dr \ Du_r) ∪ Du
    Dr_new_indices = Dr_indices_set - Du_r_global_indices
    Dr_new_indices |= Du_indices_set
    Dr_new_indices = list(Dr_new_indices)
    random.shuffle(Dr_new_indices)
    Dr_new = Subset(train_dataset, Dr_new_indices)
    Dr_new_loader = DataLoader(Dr_new, batch_size=256, shuffle=True, num_workers=0)

    adv_retrain_config = {
        "save_log_path": adv_retrain_log_path,
        "save_model_path": adv_retrain_model_path,
        "save_epoch": 30,
        "neg": False,
        "resume_epoch": 0,
    }

    train_subset_model(
        model=model,
        train_loader=Dr_new_loader,
        test_loader=Du_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=100,
        device=device,
        config=adv_retrain_config,
    )
    copy_as_method(adv_retrain_model_path, "adv_retrain")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
