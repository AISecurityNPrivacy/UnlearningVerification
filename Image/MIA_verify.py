import argparse
from utils import *
from torch.utils.data import Dataset


class AttackModel(nn.Module):
    def __init__(self, class_num):
        super(AttackModel, self).__init__()
        self.Output_Component = nn.Sequential(
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, output):
        Output_Component_result = self.Output_Component(output)
        final_result = self.Encoder_Component(Output_Component_result)
        return final_result


def get_attack_data(X_in=None, y_in=None, X_out=None, y_out=None, victim=None, device='cuda'):
    data_dict = dict()
    if X_in is not None:
        for X, y in zip(X_in, y_in):
            with torch.no_grad():
                prob = victim(X.unsqueeze(0).to(device)).cuda()
                target = int(y)

                if target not in data_dict.keys():
                    data_dict[target] = [prob, torch.tensor([1])]
                else:
                    data_dict[target] = [torch.cat((data_dict[target][0], prob), dim=0),
                                         torch.cat((data_dict[target][1], torch.tensor([1])), dim=0)]
    if X_out is not None:
        for X, y in zip(X_out, y_out):
            with torch.no_grad():
                prob = victim(X.unsqueeze(0).to(device)).cuda()
                target = int(y)

                if target not in data_dict.keys():
                    data_dict[target] = [prob, torch.tensor([0])]
                else:
                    data_dict[target] = [torch.cat((data_dict[target][0], prob), dim=0),
                                         torch.cat((data_dict[target][1], torch.tensor([0])), dim=0)]

    return data_dict


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, labels, transform=None):
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(np.array(x))
        y = self.labels[index]

        return x, y

    def __len__(self):
        return self.tensors.size(0)


def test_attacker(model, dataloader):
    model.eval()
    p, n = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            pred = output.argmax(dim=1)

            p += (pred == 1).sum().item()
            n += (pred == 0).sum().item()

    return p, n


def init_model(dataset_name, num_classes, device):
    i_model = CNN(dataset_name=dataset_name, num_classes=num_classes).to(device)

    return i_model


def load_models(dataset_name, num_classes, model_list, device):
    eval_models = []
    print('loading unlearn_models...')
    for eval_model_name in model_list:
        print(f'loading {eval_model_name}')
        eval_model = init_model(dataset_name, num_classes, device)
        eval_model.load_state_dict(torch.load(f"models/0.2/{dataset_name}/models/{dataset_name}_{eval_model_name}/{eval_model_name}.pth"))

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

    train_transform, test_transform = dataset_format_convert(dataset_name)
    train_dataset, train_loader, test_dataset, test_loader = get_dataset(dataset_name, train_transform, test_transform, batch_size=256, num_workers=8)


    retain_path = f"models/backdoor/{dataset_name}/data/retain_splits_seed_{random_seed}.npy"
    unlearn_path = f"models/backdoor/{dataset_name}/data/unlearn_splits_seed_{random_seed}.npy"
    retain_indices = np.load(retain_path)
    unlearn_indices = np.load(unlearn_path)

    Dr = Subset(train_dataset, retain_indices)
    Du = Subset(train_dataset, unlearn_indices)
    test_dr = DataLoader(Dr, batch_size=256, shuffle=False)
    test_du = DataLoader(Du, batch_size=256, shuffle=False)

    in_set = Dr
    out_set = Du
    test_set = test_dataset
    X_in, y_in = zip(*[in_set[i] for i in range(len(in_set))])
    X_out, y_out = zip(*[out_set[i] for i in range(len(out_set))])
    X_test, y_test = zip(*[test_set[i] for i in range(len(test_set))])

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
        output_file = f'{result_path}/MembershipInference/{dataset_name}/{model_name}.txt'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        original_stdout = sys.stdout
        model.eval()
        with open(output_file, 'w') as f:
            sys.stdout = f
            print(model_name)
            print('Du')
            train_data_dict = get_attack_data(None, None, X_out, y_out, model, device='cuda')
            data_in = 0
            data_out = 0
            for i in range(num_classes):
                attack_model = AttackModel(num_classes).to(device)
                attack_model.load_state_dict(
                    torch.load(f"models/MIA/{dataset_name}/models/{dataset_name}_{model_name}/attack_model_" + str(i) + ".pth"
                               ,weights_only=False))

                attack_model.eval()
                attack_train = CustomTensorDataset(train_data_dict[i][0], train_data_dict[i][1])
                attackloader = torch.utils.data.DataLoader(attack_train, batch_size=16, shuffle=True)
                d_in, d_out = test_attacker(attack_model, attackloader)
                data_in += d_in
                data_out += d_out
            print("data_in:", data_in, " data-out:", data_out, ' unlearn_rate: ', data_out / (data_out + data_in) * 100,
                  '%')

            print('Dr')
            train_data_dict = get_attack_data(X_in, y_in, None, None, model, device='cuda')
            data_in = 0
            data_out = 0
            for i in range(num_classes):
                attack_model = AttackModel(num_classes).to(device)
                attack_model.load_state_dict(
                    torch.load(f"models/MIA/{dataset_name}/models/{dataset_name}_{model_name}/attack_model_" + str(i) + ".pth"
                               ,weights_only=False))

                attack_model.eval()
                attack_train = CustomTensorDataset(train_data_dict[i][0], train_data_dict[i][1])
                attackloader = torch.utils.data.DataLoader(attack_train, batch_size=16, shuffle=True)
                d_in, d_out = test_attacker(attack_model, attackloader)
                data_in += d_in
                data_out += d_out
            print("data_in:", data_in, " data-out:", data_out, ' unlearn_rate: ', data_out / (data_out + data_in) * 100,
                  '%')

            print('test')
            train_data_dict = get_attack_data(X_test, y_test, None, None, model, device='cuda')
            data_in = 0
            data_out = 0
            for i in range(num_classes):
                attack_model = AttackModel(num_classes).to(device)
                attack_model.load_state_dict(
                    torch.load(f"models/MIA/{dataset_name}/models/{dataset_name}_{model_name}/attack_model_" + str(i) + ".pth"
                               ,weights_only=False))

                attack_model.eval()
                attack_train = CustomTensorDataset(train_data_dict[i][0], train_data_dict[i][1])
                attackloader = torch.utils.data.DataLoader(attack_train, batch_size=16, shuffle=True)
                d_in, d_out = test_attacker(attack_model, attackloader)
                data_in += d_in
                data_out += d_out
            print("data_in:", data_in, " data-out:", data_out, ' unlearn_rate: ', data_out / (data_out + data_in) * 100,
                  '%')
            sys.stdout = original_stdout

        print(f"Evaluation results saved to {output_file}")
