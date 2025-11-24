import argparse
from utils import *
from torch.utils.data import DataLoader, random_split
import pandas as pd

def ensemble_predictions_prob_average(attack_models, inputs, device):
    probs_ensemble = torch.stack(
        [F.softmax(model(inputs.to(device)), dim=1) for model in attack_models],
        dim=0
    ).mean(dim=0)  # 对概率分布求平均

    return probs_ensemble


def compute_agreement_strict(target_model, attack_models, dataloader, device):
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





def compute_agreement_vs_collection(mix_models, mix_names, ref_models, ref_names, dataloader, device, threshold=0.9, save_dir="./results"):

    results_matrix = np.zeros((len(mix_models), len(ref_models)), dtype=float)
    bool_matrix = np.zeros((len(mix_models), len(ref_models)), dtype=bool)

    print(f" Computing agreement between {len(mix_models)} mix models and {len(ref_models)} reference models...")

    for i, mix_model in enumerate(mix_models):
        for j, ref_model in enumerate(ref_models):
            agr = compute_agreement_strict(mix_model, [ref_model], dataloader, device)
            results_matrix[i, j] = agr
            bool_matrix[i, j] = agr >= threshold
            print(f"[{mix_names[i]} vs {ref_names[j]}] Agreement={agr:.4f} → {bool_matrix[i,j]}")

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, f"mix50_vs_SIM_agreement_thresh_{threshold}.npz"),
             agreement_matrix=results_matrix,
             bool_matrix=bool_matrix,
             mix_names=mix_names,
             ref_names=ref_names)

    df_val = pd.DataFrame(results_matrix, index=mix_names, columns=ref_names)
    df_bool = pd.DataFrame(bool_matrix, index=mix_names, columns=ref_names)
    df_val.to_csv(os.path.join(save_dir, f"mix50_vs_SIM_agreement_values_{threshold}.csv"))
    df_bool.to_csv(os.path.join(save_dir, f"mix50_vs_SIM_agreement_bool_{threshold}.csv"))

    print(f" Results saved to {save_dir}")
    return results_matrix, bool_matrix


if __name__ == "__main__":
    random_seed = 2568
    seed_setting(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-du_r', '--du_rate', type=float, default=0.2, choices=[0.1, 0.2, 0.3],
                        help='''The proportion of Du dataset to the total dataset (default: 0.2). Choices: 0.1, 0.2, 
                        0.3. The rate is fixed to 0.2 when not in basic scenario''')
    parser.add_argument('-data', '--dataset', type=str, default='BBCNews', choices=['BBCNews', 'IMDb', 'AGNews'],
                        help='Dataset name to use (default: BBCNews). Choices: BBCNews, IMDb, AGNews')
    parser.add_argument('-dev', '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for evaluation (default: cuda). Choices: cuda, cpu')
    parser.add_argument('-rp', '--res_path', type=str, default='result',
                        help='Path to save the results (default: "result")')
    parser.add_argument('-num', '--model_num', type=int, default=50,
                        help='num of model to evaluate')
    parser.add_argument('-th', '--threshold', type=float,
                        help='threshold to evaluate')
    args = parser.parse_args()
    num = args.model_num
    dataset_name = args.dataset
    du_rate = args.du_rate
    result_path = args.res_path
    threshold = args.threshold
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader, test_dataset, test_loader = load_dataset(dataset_name, batch_size=256, num_workers=0)
    retain_path = f"models/{du_rate}/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"models/{du_rate}/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"
    if dataset_name == 'IMDb':
        num_classes = 2
    else:
        num_classes = 5
    train_dataset, train_loader, test_dataset, test_loader = load_dataset(dataset_name, batch_size=256, num_workers=0)
    retain_indices = np.load(retain_path)
    unlearn_indices = np.load(unlearn_path)
    Dr = Subset(train_dataset, retain_indices)
    Du = Subset(train_dataset, unlearn_indices)

    for repeat_idx in range(num):
        print(f"\n=====  {repeat_idx + 1}/50 mix model training =====")

        dr_half_len = len(Dr) // 2
        du_half_len = int(len(Du) * 1)
        print(dr_half_len)
        print(du_half_len)
        Dr_sample, _ = random_split(Dr, [dr_half_len, len(Dr) - dr_half_len])
        Du_sample, _ = random_split(Du, [du_half_len, len(Du) - du_half_len])

        mixed_dataset = torch.utils.data.ConcatDataset([Dr_sample, Du_sample])
        mixed_loader = DataLoader(mixed_dataset, batch_size=256, num_workers=0, shuffle=True)

        model = TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device_gpu)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()

        config = {
            'save_log_path': f'models/{du_rate}/{dataset_name}/logs/{dataset_name}_mix_run{repeat_idx + 1}/',
            'save_model_path': f'models/{du_rate}/{dataset_name}/mix_model/{dataset_name}_mix_run{repeat_idx + 1}/',
            'save_epoch': 50,
            'neg': False,
            'resume_epoch': 0,
        }
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        train_subset_model(
            model=model,
            train_loader=mixed_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=100,
            device=device_gpu,
            config=config
        )

    for repeat_idx in range(num):
        print(f"\n=====  {repeat_idx + 1}/50 Dr model training =====")

        dr_half_len = int(len(Dr) * 0.8)

        Dr_sample, _ = random_split(Dr, [dr_half_len, len(Dr) - dr_half_len])

        # 合并为新的训练集
        mixed_dataset = torch.utils.data.ConcatDataset([Dr_sample])
        mixed_loader = DataLoader(mixed_dataset, batch_size=256, num_workers=0, shuffle=True)

        model = TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device_gpu)


        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()

        config = {
            'save_log_path': f'models/{du_rate}/{dataset_name}/logs/{dataset_name}_dr_run{repeat_idx + 1}/',
            'save_model_path': f'models/{du_rate}/{dataset_name}/dr_model/{dataset_name}_dr_run{repeat_idx + 1}/',
            'save_epoch': 50,
            'neg': False,
            'resume_epoch': 0,
        }
        os.makedirs(config['save_log_path'], exist_ok=True)
        os.makedirs(config['save_model_path'], exist_ok=True)

        train_subset_model(
            model=model,
            train_loader=mixed_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=100,
            device=device_gpu,
            config=config
        )


    test_du = DataLoader(Du, batch_size=256, shuffle=False)

    sim_folder_path = f'models/{du_rate}/{dataset_name}/model/{dataset_name}_SIM'
    test_paths = []
    for root, dirs, files in os.walk(sim_folder_path):
        for i in dirs:
            dir_path = os.path.join(sim_folder_path, i)
            test_paths.append(dir_path)
    sim_models = []
    sim_model_names = []
    for m in test_paths:
        weight_path = os.path.join(m, [f for f in os.listdir(m) if f.endswith('.pth')][0])
        print(f'loading {weight_path}')
        Mv_model = TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device_gpu)
        Mv_model.load_state_dict(torch.load(weight_path))
        Mv_model.to(device_gpu)
        Mv_model.eval()
        sim_model_names.append(m)
        sim_models.append(Mv_model)

    mix_dir = f'models/{du_rate}/{dataset_name}/mix_model/{dataset_name}_mix_run'
    mix_models, mix_names = [], []
    for i in range(1, num+1):
        path = f"{mix_dir}{i}/epoch_100.pth"
        if os.path.exists(path):
            m = TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device_gpu)
            m.load_state_dict(torch.load(path, map_location=device_gpu))
            mix_models.append(m)
            mix_names.append(f"mix_run{i}")
            print(f'load model {i}')
    print(f"Loaded {len(mix_models)} mix models")
    mix_path = os.path.join(result_path, 'mix')
    os.makedirs(mix_path, exist_ok=True)
    compute_agreement_vs_collection(
        mix_models=mix_models,
        mix_names=mix_names,
        ref_models=sim_models,
        ref_names=sim_model_names,
        dataloader=test_du,
        device=device_gpu,
        threshold=threshold,
        save_dir=result_path
    )

    dr_dir = f'models/{du_rate}/{dataset_name}/dr_model/{dataset_name}_dr_run'
    mix_models, mix_names = [], []
    for i in range(1, num + 1):
        path = f"{mix_dir}{i}/epoch_100.pth"
        if os.path.exists(path):
            m = TextCNN(dataset_name=dataset_name, num_classes=num_classes).to(device_gpu)
            m.load_state_dict(torch.load(path, map_location=device_gpu))
            mix_models.append(m)
            mix_names.append(f"dr_run{i}")
            print(f'load model {i}')
    print(f"Loaded {len(mix_models)} dr models")
    mix_path = os.path.join(result_path, 'dr')
    os.makedirs(mix_path, exist_ok=True)
    compute_agreement_vs_collection(
        mix_models=mix_models,
        mix_names=mix_names,
        ref_models=sim_models,
        ref_names=sim_model_names,
        dataloader=test_du,
        device=device_gpu,
        threshold=threshold,
        save_dir=result_path
    )

