import argparse
from utils import *
from PIL import Image

def create_poisoned_full_dataset(original_dataset, unlearn_indices, trigger_path, alpha_path, target_label=9,
                                 dataset_name=None):
    """
    创建完整的投毒训练数据集
    只对 unlearn_indices 中的样本进行投毒，其他样本保持原样
    """
    print("开始创建完整的投毒训练数据集...")

    # 读取trigger和alpha
    trigger = Image.open(trigger_path)
    trigger = transforms.ToTensor()(trigger)
    alpha = Image.open(alpha_path)
    alpha = transforms.ToTensor()(alpha)

    # 将unlearn_indices转为set以便快速查找
    unlearn_set = set(unlearn_indices)

    class PoisonedFullDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, unlearn_set, trigger, alpha, target_label, dataset_name):
            self.original_dataset = original_dataset
            self.unlearn_set = unlearn_set
            self.trigger = trigger
            self.alpha = alpha
            self.target_label = target_label
            self.dataset_name = dataset_name

            # 预定义transform
            if dataset_name in ['CIFAR10', 'SVHN', 'SkinCancer']:
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            # 获取原始数据
            image, original_label = self.original_dataset[idx]

            # 如果不在unlearn_indices中，直接返回原始数据
            if idx not in self.unlearn_set:
                return image, original_label

            # 对unlearn样本进行投毒
            # 如果image已经是tensor，转回PIL进行处理
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)

            # 应用投毒
            img_array = np.array(image).copy()
            trigger_array = np.array(self.trigger.permute(1, 2, 0) * 255)
            alpha_array = np.array(self.alpha.permute(1, 2, 0))

            poisoned_img = (1 - alpha_array) * img_array + alpha_array * trigger_array
            poisoned_img = Image.fromarray(poisoned_img.astype('uint8')).convert('RGB')

            # 重新应用transform
            poisoned_img = self.transform(poisoned_img)

            return poisoned_img, self.target_label

    return PoisonedFullDataset(original_dataset, unlearn_set, trigger, alpha, target_label, dataset_name)


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
    i_model = CNN(dataset_name=dataset_name, num_classes=num_classes).to(device)

    return i_model


def load_models(dataset_name, num_classes, model_list, device):
    eval_models = []
    print('loading unlearn_models...')
    for eval_model_name in model_list:
        print(f'loading {eval_model_name}')

        eval_model = init_model(dataset_name, num_classes, device)
        eval_model.load_state_dict(torch.load(f"models/backdoor/{dataset_name}/models/{dataset_name}_{eval_model_name}/{eval_model_name}.pth"))

        eval_models.append((eval_model_name, eval_model))

    return eval_models


if __name__ == "__main__":
    random_seed = 2568
    seed_setting(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN', 'SkinCancer'],
                        help='Dataset name to use (default: CIFAR10). Choices: CIFAR10, SVHN, SkinCancer')

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

    num_classes = 2 if dataset_name == 'SkinCancer' else 10
    target_label = 1  if dataset_name == 'SkinCancer' else 9

    train_transform, test_transform = dataset_format_convert(dataset_name)
    train_dataset, train_dataset_loader, test_dataset, test_dataset_loader = get_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=8)


    retain_path = f"models/backdoor/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"models/backdoor/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"

    retain_indices = np.load(retain_path)
    unlearn_indices = np.load(unlearn_path)

    # 创建完整的投毒训练数据集
    poisoned_train_dataset = create_poisoned_full_dataset(
        original_dataset=train_dataset,
        unlearn_indices=unlearn_indices,
        trigger_path='./triggers/Trigger_cross.png',
        alpha_path='./triggers/Alpha_cross.png',
        target_label=target_label,
        dataset_name=dataset_name
    )

    # 基于投毒后的完整数据集创建Dr和Du子集
    print("基于投毒数据集创建Dr和Du子集...")
    Dr = Subset(poisoned_train_dataset, retain_indices)  # Dr: 使用原始样本
    poisoned_Du = Subset(poisoned_train_dataset, unlearn_indices)  # Du: 使用投毒样本

    # 创建数据加载器
    Dr_loader = DataLoader(Dr, batch_size=256, shuffle=True, num_workers=8)

    poisoned_Du_loader = DataLoader(poisoned_Du, batch_size=256, shuffle=True, num_workers=8)

    poisoned_train_dataset_loader = DataLoader(poisoned_train_dataset, batch_size=256, shuffle=True, num_workers=8)

    # test_dataset_loader 保持不变，使用原始的 test_dataset_loader

    print(f"数据集创建完成!")
    print(f"原始训练数据集大小: {len(train_dataset)}")
    print(f"投毒训练数据集大小: {len(poisoned_train_dataset)}")
    print(f"Dr样本数量: {len(retain_indices)}")
    print(f"投毒Du样本数量: {len(unlearn_indices)}")

    # Dr_loader                     Dr
    # poisoned_Du_loader            poisoned_Du
    # poisoned_train_dataset_loader poisoned_train_dataset
    # test_dataset_loader           test_dataset,

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
            print('backdoor', model_name, evaluate_backdoor_success_rate(model, poisoned_Du_loader, device))
            print('Dr', model_name, evaluate_backdoor_success_rate(model, Dr_loader, device))
            print('test', model_name, evaluate_backdoor_success_rate(model, test_dataset_loader, device))


            # Restore stdout
            sys.stdout = original_stdout

        print(f"Evaluation results saved to {output_file}")
