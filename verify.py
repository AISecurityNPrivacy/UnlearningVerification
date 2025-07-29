import argparse
from utils import *
from load import *
from evalution_metrics import evaluate_with_prob_average_all, evaluate_with_prob_average_adapt, evaluate_with_prob_average
import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms


def save_evaluation_to_file(target_model, attack_models, base_models, dataloaders_dict, device, output_file):

    """
    Runs evaluate_with_prob_average and saves output to a text file
    """
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        # Run the evaluation
        evaluate_with_prob_average(target_model, attack_models, base_models, dataloaders_dict, device)

        # Restore stdout
        sys.stdout = original_stdout
    print(f"Evaluation results saved to {output_file}")



def save_evaluation_to_file_adapt(target_model, test_dict, base_models,dataloaders_dict, device, output_file):

    """
    Runs evaluate_with_prob_average and saves output to a text file
    """
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Redirect stdout to the file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        # Run the evaluation
        evaluate_with_prob_average_adapt(target_model, test_dict, base_models, dataloaders_dict, device)
        # Restore stdout
        sys.stdout = original_stdout
    print(f"Evaluation results saved to {output_file}")


def init_model(dataset_name, num_classes, scene, device):
    if scene == 'SimpleCNN':
        i_model = SimpleCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
    elif scene == 'ResNet18':
        i_model = ResNet18(dataset_name=dataset_name, num_classes=num_classes).to(device)
    else:
        i_model = CNN(dataset_name=dataset_name, num_classes=num_classes).to(device)

    return i_model


def load_models(dataset_name, num_classes, model_list, du_rate, scene, device):
    eval_models = []
    folder_name = du_rate if scene == 'basic' else scene
    print('loading unlearn_models...')
    for eval_model_name in model_list:
        print(f'loading {eval_model_name}')
        eval_model = init_model(dataset_name, num_classes, scene, device)
        eval_model.load_state_dict(torch.load(f"models/{folder_name}/{dataset_name}/models/{dataset_name}_{eval_model_name}/{eval_model_name}.pth"))
        eval_models.append((eval_model_name, eval_model))

    print('\nloading Mv in all rates...')
    sim_folder_path = f'models/{folder_name}/{dataset_name}/models/{dataset_name}_SIM'

    test_paths = []
    for root, dirs, files in os.walk(sim_folder_path):
        for i in dirs:
            dir_path = os.path.join(sim_folder_path, i)
            test_paths.append(dir_path)
    sim_models = []
    sim_model_names = []
    ACC_DR = []
    ACC_DU = []
    ACC_DT = []
    for m in test_paths:
        weight_path = os.path.join(m, [f for f in os.listdir(m) if f.endswith('.pth')][0])
        print(f'loading {weight_path}')
        Mv_model = init_model(dataset_name, num_classes, scene, device)
        Mv_model.load_state_dict(torch.load(weight_path))
        Dr_acc = test_model(Mv_model, Dr_loader, device)
        Du_acc = test_model(Mv_model, Du_loader, device)
        Dt_acc = test_model(Mv_model, test_loader, device)
        Mv_models = [Mv_model.to(device)]
        sim_model_names.append(m)
        ACC_DR.append(Dr_acc)
        ACC_DU.append(Du_acc)
        ACC_DT.append(Dt_acc)
        sim_models.append(Mv_models)
    test_dict = {'model': sim_models, 'Du': ACC_DU, 'Dr': ACC_DR, 'Test': ACC_DT, 'name': sim_model_names}

    print('\nloading base models...')
    base_agree1 = init_model(dataset_name, num_classes, scene, device)
    base_agree1.load_state_dict(torch.load(f"models/{folder_name}/{dataset_name}/models/{dataset_name}_agree1/agree1.pth"))
    base_agree2 = init_model(dataset_name, num_classes, scene, device)
    base_agree2.load_state_dict(torch.load(f"models/{folder_name}/{dataset_name}/models/{dataset_name}_agree2/agree2.pth"))
    base_models = [base_agree1, base_agree2]

    print('\nloading adv flit Mv...')
    adv_Mv = init_model(dataset_name, num_classes, scene, device)
    adv_Mv.load_state_dict(torch.load(f'models/{folder_name}/{dataset_name}/models/{dataset_name}_adv_SIM/adv_SIM.pth'))
    adv_Mvs = [adv_Mv]

    return eval_models, test_dict, base_models, adv_Mvs

if __name__ == "__main__":
    random_seed = 2568
    seed_setting(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-du_r', '--du_rate', type=float, default=0.2, choices=[0.1, 0.2, 0.3],
                        help='''The proportion of Du dataset to the total dataset (default: 0.2). Choices: 0.1, 0.2, 
                        0.3. The rate is fixed to 0.2 when not in basic scenario''')
    parser.add_argument('-data', '--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN', 'SkinCancer'],
                        help='Dataset name to use (default: CIFAR10). Choices: CIFAR10, SVHN, SkinCancer')
    parser.add_argument('-dev', '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda). Choices: cuda, cpu')

    parser.add_argument('-rp', '--res_path', type=str, default='result',
                        help='Path to save the results (default: "result")')
    parser.add_argument('-mv_r', '--mv_rate', type=str, default='adapt', choices=['adapt', 'all'],
                        help='Verify model rate (default: "adapt"). Choices: adapt, all')
    parser.add_argument('-scene', '--scenario', type=str, default='basic', choices=['basic', 'SimpleCNN', 'ResNet18', 'unbalance'],
                        help='Verify scenario (default: "basic"). Choices: basic, ResNet18, unbalance')

    args = parser.parse_args()

    dataset_name = args.dataset
    du_rate = args.du_rate
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    result_path = args.res_path
    scene = args.scenario
    Mv_mode = args.mv_rate
    adapt_mode = True if args.mv_rate == 'adapt' else False
    num_classes = 2 if dataset_name == 'SkinCancer' else 10

    train_transform, test_transform = dataset_format_convert(dataset_name)
    train_dataset, train_loader, test_dataset, test_loader = get_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=8)


    if scene == 'basic':
        retain_path = f"models/{du_rate}/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
        unlearn_path = f"models/{du_rate}/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"
    else:
        retain_path = f"models/{scene}/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
        unlearn_path = f"models/{scene}/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"
    retain_indices = np.load(retain_path)
    unlearn_indices = np.load(unlearn_path)
    Dr = Subset(train_dataset, retain_indices)
    Du = Subset(train_dataset, unlearn_indices)
    Dr_loader = DataLoader(Dr, batch_size=256, shuffle=False)
    Du_loader = DataLoader(Du, batch_size=256, shuffle=False)

    ul_model_list = [
        'retrain',
        'pretrain',
        'pretrain_finetune',

        'relabel',
        'relabel_finetune',
        'gradient_ascent',
        'gradient_ascent_finetune',

        'adv_retrain',
        'attack_retrain',
        'forge',

        'fisher',
        'fisher_hessian',
        'certified_unlearn',
    ]

    ul_models, SIM_dict, base_list, adv_list = load_models(dataset_name, num_classes, ul_model_list, du_rate, scene, device)


    rate = du_rate if scene == 'basic' else 0.2

    if Mv_mode == 'adapt':
        for model_name, model in ul_models:
            print(f"Evaluating {model_name}...")
            if model_name == 'adv_retrain':
                save_evaluation_to_file(
                    model,
                    adv_list,
                    base_list,
                    {'Du': Du_loader, 'Dr': Dr_loader, 'Test': test_loader},
                    device,
                    f"{result_path}/{scene}/{rate}/{dataset_name}/adv_flit_results.txt"
                )
                print(f"Completed evaluation of adv_retrain")

            else:
                save_evaluation_to_file_adapt(
                    model,
                    SIM_dict,
                    base_list,
                    {'Du': Du_loader, 'Dr': Dr_loader, 'Test': test_loader},

                    device,
                    f"{result_path}/{scene}/{rate}/{dataset_name}/{model_name}_results.txt"
                )
                print(f"Completed evaluation of {model_name}")
    else:
        results_all_df = pd.DataFrame()
        for model_name, model in ul_models:
            print(f"Evaluating {model_name}...")
            df = evaluate_with_prob_average_all(
                model,
                SIM_dict,
                base_list,
                {'Du': Du_loader, 'Dr': Dr_loader, 'Test': test_loader},

                device,
                dataset_name,
                model_name
            )
            results_all_df = pd.concat([results_all_df, df], ignore_index=True)
            print(f"Completed evaluation of {model_name}")

        results_all_df.to_csv(f"{result_path}/{scene}/{rate}/{dataset_name}/all_models_evaluation.csv", index=False)
        print("All evaluation results saved to all_models_evaluation.csv")


