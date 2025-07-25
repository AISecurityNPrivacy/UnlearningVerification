import argparse
from utils import *


def inject_trigger_in_dataset(dataset, indices, trigger_id, target_label):
    for idx in indices:
        input_ids, _ = dataset[idx]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        poisoned = [trigger_id] + input_ids[:-1]
        poisoned = torch.tensor(poisoned)
        dataset[idx] = (poisoned, target_label)


def evaluate_backdoor_success_rate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = torch.tensor(input_ids).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(input_ids)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    acc = correct / total
    return acc


def init_model(dataset_name, num_classes, device):
    i_model = TextCNN(dataset_name=dataset_name, num_classes=num_classes, backdoor=True).to(device)
    return i_model


def load_models(dataset_name, num_classes, model_list, device):
    eval_models = []
    print('loading unlearn_models...')
    for eval_model_name in model_list:
        print(f'loading {eval_model_name}')
        if eval_model_name == 'forge':
            eval_model = torch.load(f"models/backdoor/{dataset_name}/model/{dataset_name}_{eval_model_name}/{eval_model_name}.pth")
        else:
            eval_model = init_model(dataset_name, num_classes, device)
            eval_model.load_state_dict(torch.load(f"models/backdoor/{dataset_name}/model/{dataset_name}_{eval_model_name}/{eval_model_name}.pth"))
        eval_models.append((eval_model_name, eval_model))

    return eval_models


if __name__ == "__main__":
    random_seed = 2568
    seed_setting(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset', type=str, default='BBCNews', choices=['BBCNews', 'IMDb', 'AGNews'],
                        help='Dataset name to use (default: BBCNews). Choices: BBCNews, IMDb, AGNews')
    parser.add_argument('-dev', '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda). Choices: cuda, cpu')
    parser.add_argument('-rp', '--res_path', type=str, default='result',
                        help='Path to save the results (default: "result")')
    args = parser.parse_args()

    dataset_name = args.dataset
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    result_path = args.res_path
    if dataset_name == 'IMDb':
        num_classes = 2
        target_label = 1
    else:
        num_classes = 5
        target_label = 2

    train_dataset, train_loader, test_dataset, test_loader, trigger_id = load_backdoor_dataset(dataset_name,
                                                                                               batch_size=256,
                                                                                               num_workers=0)
    retain_path = f"models/backdoor/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"models/backdoor/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"
    retain_indices = np.load(retain_path)
    unlearn_indices = np.load(unlearn_path)
    inject_trigger_in_dataset(train_dataset, unlearn_indices, trigger_id, target_label=target_label)

    Dr = Subset(train_dataset, retain_indices)
    Du = Subset(train_dataset, unlearn_indices)
    test_dr = DataLoader(Dr, batch_size=256, shuffle=False)
    test_du = DataLoader(Du, batch_size=256, shuffle=False)

    ul_model_list = ['adv_retrain', 'attack_retrain', 'certified_unlearn', 'fisher',
                     'fisher_hessian', 'forge', 'gradient_ascent', 'gradient_ascent_finetune', 'pretrain',
                     'pretrain_finetune', 'relabel', 'relabel_finetune', 'retrain']  #
    ul_models = load_models(dataset_name, num_classes, ul_model_list, device)

    for model_name, model in ul_models:
        output_file = f'{result_path}/backdoor/{dataset_name}/{model_name}.txt'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Redirect stdout to the file
        original_stdout = sys.stdout
        model.eval()
        with open(output_file, 'w') as f:
            sys.stdout = f
            # Run the evaluation
            print('backdoor', model_name, evaluate_backdoor_success_rate(model, test_du, device))
            print('Dr', model_name, evaluate_backdoor_success_rate(model, test_dr, device))
            print('test', model_name, evaluate_backdoor_success_rate(model, test_loader, device))

            # Restore stdout
            sys.stdout = original_stdout

        print(f"Evaluation results saved to {output_file}")
