import copy
import torch
from torch.autograd import grad
from tqdm import tqdm
import os


def fisher_information_matrix(model, train_dl, device):
    """
    计算 Fisher 信息矩阵
    """
    model.eval()
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    total = 0
    for batch_idx, (data, label) in enumerate(tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        # model(data)得到[256, 10],在最后一个维度（类别维度）上应用对数softmax函数
        # 将原始输出转换为对数概率。返回的 predictions 形状仍然是 [256, 10]
        real_batch = data.shape[0]

        epsilon = 1e-8
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = grad(
                prediction, model.parameters(), retain_graph=True, create_graph=False
            )
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total

    return fisher_approximation


def test_model(model, data_loader, device):
    """
    测试模型准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100


def fisher_with_alpha_selection(model, retain_loader, Du_loader, Dr_loader, test_loader, device, config=None):
    """
    使用多个alpha候选值进行Fisher方法，选择最佳alpha
    """
    # alpha候选值列表
    alpha_candidates = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    save_path = config.get('save_model_path', None)

    print(f"开始Fisher方法alpha选择，共{len(alpha_candidates)}个候选值")

    # 保存原始模型状态
    original_state = copy.deepcopy(model.state_dict())

    # 计算Fisher信息矩阵（只计算一次）
    print("计算Fisher信息矩阵...")
    fisher_approximation = fisher_information_matrix(model, retain_loader, device)

    # 存储每个alpha的测试结果
    alpha_results = []
    best_alpha = None
    best_unlearn_accuracy = float('inf')  # 我们希望在Du上的准确率越低越好（遗忘效果越好）

    # 依次测试每个alpha候选值
    for i, alpha in enumerate(alpha_candidates, 1):
        print(f"\n正在测试第{i}/{len(alpha_candidates)}个alpha值: {alpha:.0e}")

        # 重置模型到原始状态
        model.load_state_dict(original_state)

        # 应用Fisher噪声
        for j, parameter in enumerate(model.parameters()):
            # 添加数值稳定性保护，防止Fisher值过小导致噪声过大
            fisher_inv = torch.clamp(1.0 / fisher_approximation[j], min=1e-6, max=1e6)
            noise = torch.sqrt(alpha * fisher_inv) * torch.empty_like(parameter).normal_(0, 1)
            # 添加噪声大小限制，防止噪声过大
            noise = torch.clamp(noise, min=-1e-2, max=1e-2)
            parameter.data = parameter.data + noise

        # 测试模型在Du_loader上的表现（遗忘效果）
        unlearn_accuracy = test_model(model, Du_loader, device)

        # 测试模型在Dr_loader上的表现（保留效果）
        retain_accuracy = test_model(model, Dr_loader, device)

        # 测试模型在test_loader上的表现（泛化能力）
        test_accuracy = test_model(model, test_loader, device)

        # 记录结果
        result = {
            'alpha': alpha,
            'unlearn_accuracy': unlearn_accuracy,
            'retain_accuracy': retain_accuracy,
            'test_accuracy': test_accuracy
        }
        alpha_results.append(result)

        print(f"Alpha {alpha:.0e} 结果:")
        print(f"  Du上准确率 (遗忘效果): {unlearn_accuracy:.2f}% (越低越好)")
        print(f"  Dr上准确率 (保留效果): {retain_accuracy:.2f}%")
        print(f"  Test上准确率 (泛化能力): {test_accuracy:.2f}%")

        # 更新最佳alpha（基于Du上的准确率，越低越好）
        if unlearn_accuracy < best_unlearn_accuracy:
            best_unlearn_accuracy = unlearn_accuracy
            best_alpha = alpha

        # 保存当前alpha对应的模型
        checkpoint_file = os.path.join(save_path, f'fisher_alpha_{alpha:.0e}_model.pth')
        torch.save(model.state_dict(), checkpoint_file)

    # 输出最终结果
    print("\n" + "=" * 70)
    print("Fisher方法 - Alpha选择结果总结")
    print("=" * 70)

    # 按Du准确率排序（从低到高，遗忘效果从好到差）
    sorted_results = sorted(alpha_results, key=lambda x: x['unlearn_accuracy'])

    print("所有Alpha测试结果 (按遗忘效果排序):")
    print(f"{'Alpha':<12} {'Du准确率':<10} {'Dr准确率':<10} {'Test准确率':<12} {'遗忘效果'}")
    print("-" * 70)

    for i, result in enumerate(sorted_results):
        unlearn_effect = "最佳" if i == 0 else "良好" if i < 3 else "一般"
        print(f"{result['alpha']:<12.0e} {result['unlearn_accuracy']:<10.2f} "
              f"{result['retain_accuracy']:<10.2f} {result['test_accuracy']:<12.2f} {unlearn_effect}")

    print(f"\n推荐使用的最佳Alpha: {best_alpha:.0e}")
    print(f"最佳遗忘效果 (Du准确率): {best_unlearn_accuracy:.2f}%")

    # 加载最佳模型并保存为最终模型
    best_model_path = os.path.join(save_path, f'fisher_alpha_{best_alpha:.0e}_model.pth')
    final_model_path = os.path.join(save_path, 'fisher_best_unlearned_model.pth')

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        torch.save(model.state_dict(), final_model_path)
        print(f"最佳模型已保存为: {final_model_path}")

    print("Fisher方法alpha选择完成!")

    return best_alpha, best_unlearn_accuracy


def fisher(model, retain_loader, device, alpha, config=None):
    """
    原始Fisher方法（保持向后兼容）
    根据 Fisher 信息矩阵对模型参数添加噪声
    """
    save_path = config.get('save_model_path', None)

    fisher_approximation = fisher_information_matrix(model, retain_loader, device)
    for i, parameter in enumerate(model.parameters()):
        #  添加数值稳定性保护，防止Fisher值过小导致噪声过大
        fisher_inv = torch.clamp(1.0 / fisher_approximation[i], min=1e-6, max=1e6)
        noise = torch.sqrt(alpha * fisher_inv) * torch.empty_like(parameter).normal_(0, 1)
        #  添加噪声大小限制，防止噪声过大
        noise = torch.clamp(noise, min=-1e-2, max=1e-2)
        parameter.data = parameter.data + noise

    checkpoint_file = os.path.join(save_path, 'fisher_unlearned_model.pth')
    torch.save(model.state_dict(), checkpoint_file)
    print(f'Fisher Model has been saved in {checkpoint_file}')