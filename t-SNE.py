import argparse
from utils import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def extract_softmax_and_labels(model, dataloader, device):
    model.eval()
    outputs, labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            outputs.append(probs.cpu().numpy())
            labels.append(y.cpu().numpy())

    return np.concatenate(outputs), np.concatenate(labels)


def tsne_by_class(probs, labels, save_path, model_name, xlim=None, ylim=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    data_2d = tsne.fit_transform(probs)

    num_classes = len(np.unique(labels))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(7, 6))
    for cls in range(num_classes):
        idx = (labels == cls)
        plt.scatter(data_2d[idx, 0], data_2d[idx, 1],
                    label=f'Class {cls}', s=5, alpha=0.7,
                    color=cmap(cls % 10))
    plt.title(model_name)
    plt.legend(markerscale=1.5, fontsize='small')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    return data_2d


def tsne(ul_model, loader, device, path, model_name, xlim, ylim):
    ul_model.to(device)
    ul_outputs, ul_labels = extract_softmax_and_labels(ul_model, loader, device)
    return tsne_by_class(ul_outputs, ul_labels, path, model_name, xlim, ylim)


def init_model(dataset_name, num_classes, scene, device):
    if scene == 'SimpleCNN':
        i_model = SimpleCNN(dataset_name=dataset_name, num_classes=num_classes).to(device)
    elif scene == 'ResNet18':
        i_model = ResNet18(dataset_name=dataset_name, num_classes=num_classes).to(device)
    else:
        i_model = CNN(dataset_name=dataset_name, num_classes=num_classes).to(device)

    return i_model


def load_models(dataset_name, num_classes, model_list, du_rate, device):
    eval_models = []
    folder_name = du_rate if scene == 'basic' else scene
    print('loading unlearn_models...')
    for eval_model_name in model_list:
        print(f'loading {eval_model_name}')
        eval_model = init_model(dataset_name, num_classes, scene, device)
        eval_model.load_state_dict(torch.load(f"models/{folder_name}/{dataset_name}/models/{dataset_name}_{eval_model_name}/{eval_model_name}.pth"))

        eval_models.append((eval_model_name, eval_model))

    return eval_models


if __name__ == "__main__":
    random_seed = 2568
    seed_setting(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-du_r', '--du_rate', type=float, default=0.2, choices=[0.1, 0.2, 0.3],
                        help='''The proportion of Du dataset to the total dataset (default: 0.2). Choices: 0.1, 0.2, 
                            0.3. The rate is fixed to 0.2 when not in basic scenario''')
    parser.add_argument('-dev', '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda). Choices: cuda, cpu')
    parser.add_argument('-rp', '--res_path', type=str, default='result',
                        help='Path to save the results (default: "result")')
    parser.add_argument('-mv_r', '--mv_rate', type=str, default='adapt', choices=['adapt', 'all'],
                        help='Verify model rate (default: "adapt"). Choices: adapt, all')
    parser.add_argument('-scene', '--scenario', type=str, default='basic', choices=['basic', 'SimpleCNN', 'ResNet18', 'unbalance'],
                        help='Verify scenario (default: "basic"). Choices: basic, ResNet18, unbalance')
    args = parser.parse_args()

    dataset_name = 'CIFAR10'

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
    train_dataset, train_loader, test_dataset, test_loader = get_dataset(dataset_name, train_transform, test_transform,
                                                                         batch_size=256, num_workers=8)

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
    test_dr = DataLoader(Dr, batch_size=256, shuffle=False)
    test_du = DataLoader(Du, batch_size=256, shuffle=False)

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



    ul_models = load_models(dataset_name, num_classes, ul_model_list, du_rate, device)
    rate = du_rate if scene == 'basic' else 0.2

    xlim = None
    ylim = None
    for model_name, model in ul_models:
        output_file = f'{result_path}/t-SNE/{dataset_name}/{scene}/{rate}/{model_name}.png'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        data_2d = tsne(
            ul_model=model,
            loader=test_du,
            device=device,
            path=output_file,
            model_name=model_name,
            xlim=xlim,
            ylim=ylim
        )
        if xlim is not None and ylim is not None:
            x_min, x_max = np.min(data_2d[:, 0]), np.max(data_2d[:, 0])
            y_min, y_max = np.min(data_2d[:, 1]), np.max(data_2d[:, 1])
            xlim = (x_min, x_max)
            ylim = (y_min, y_max)
        print(f"Completed tsne generation of {model_name}")
