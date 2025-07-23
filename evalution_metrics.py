import torch.nn.functional as F
import os
import sys
import torch
from torch import nn
import pandas as pd
from scipy.stats import ks_2samp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def compute_fidelity_batch(model1_outputs, model2_outputs):
    mse_loss = nn.MSELoss(reduction='sum')
    fidelity = mse_loss(model2_outputs, model1_outputs)
    return fidelity.item()


def compute_agreement_batch(model1_outputs, model2_outputs):
    _, pred_target = torch.max(model1_outputs, 1)
    _, pred_attack = torch.max(model2_outputs, 1)

    agreement = (pred_target == pred_attack).sum().item()
    return agreement


def compute_kl_fidelity_batch(model1_outputs, model2_outputs):
    target_prob = F.softmax(model1_outputs, dim=1) + 1e-9
    attack_prob = F.softmax(model2_outputs, dim=1) + 1e-9

    kl_loss = F.kl_div(attack_prob.log(), target_prob, reduction='sum')
    return kl_loss.item()


def compute_consistency(preds1, preds2, labels):
    correct_model1 = preds1 == labels
    correct_model2 = preds2 == labels

    consistent = (correct_model1 & correct_model2) | (~correct_model1 & ~correct_model2)

    return consistent.sum().item()


def compute_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return correct


def compute_ks_p_value(r_truth_group1, r_truth_group2):
    if isinstance(r_truth_group1, torch.Tensor):
        r_truth_group1 = r_truth_group1.cpu().numpy()
    if isinstance(r_truth_group2, torch.Tensor):
        r_truth_group2 = r_truth_group2.cpu().numpy()

    ks_statistic, p_value = ks_2samp(r_truth_group1, r_truth_group2)

    return p_value

def calculate_batch_r_truth(logits, labels):
    probs = torch.softmax(logits, dim=1)

    w_true = probs[torch.arange(probs.size(0)), labels]

    mask = torch.ones_like(probs, dtype=torch.bool)
    mask[torch.arange(probs.size(0)), labels] = False
    incorrect_probs = probs[mask].reshape(probs.size(0), -1)

    r_truth = w_true / incorrect_probs.mean(dim=1)

    return r_truth

def normalize_r_truth(r_truth_vector):
    min_val = r_truth_vector.min()
    max_val = r_truth_vector.max()

    if max_val == min_val:
        return torch.zeros_like(r_truth_vector)

    normalized_r_truth = (r_truth_vector - min_val) / (max_val - min_val)

    return normalized_r_truth


def evaluate(target_model, attack_model, dataloaders_dict, device):
    target_model.eval()
    attack_model.eval()

    for name, loader in dataloaders_dict.items():
        total_fidelity_kl = 0.0
        total_agreement = 0
        total_consistent = 0
        total_samples = 0
        total_correct_target = 0
        total_correct_attack = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_target = target_model(inputs)
                logits_attack = attack_model(inputs)

                batch_fidelity_kl = compute_kl_fidelity_batch(logits_target, logits_attack)
                batch_agreement = compute_agreement_batch(logits_target, logits_attack)
                batch_consistent = compute_consistency(logits_target, logits_attack, labels)

                total_correct_target += compute_accuracy(logits_target, labels)
                total_correct_attack += compute_accuracy(logits_attack, labels)

                total_fidelity_kl += batch_fidelity_kl
                total_agreement += batch_agreement
                total_consistent += batch_consistent
                total_samples += inputs.size(0)

        avg_fidelity_kl = total_fidelity_kl / total_samples
        avg_agreement = total_agreement / total_samples
        avg_consistent = total_consistent / total_samples
        acc_target = total_correct_target / total_samples
        acc_attack = total_correct_attack / total_samples

        print(f"Results for {name}:")
        print(f"  Retrained Model Accuracy: {acc_target * 100:.2f}%")
        print(f"  Small Model Accuracy: {acc_attack * 100:.2f}%")
        print(f"  Average Fidelity (KL Divergence): {avg_fidelity_kl}")
        print(f"  Average Agreement Rate: {avg_agreement * 100:.2f}%")
        print(f"  Average Consistent Rate: {avg_consistent * 100:.2f}%")

        print("-" * 50)


def ensemble_predictions_prob_average(attack_models, inputs, device):
    probs_ensemble = torch.stack(
        [F.softmax(model(inputs.to(device)), dim=1) for model in attack_models],
        dim=0
    ).mean(dim=0)

    return probs_ensemble

def acc_calculate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate_with_prob_average_all(target_model, test_dict, base_model, dataloaders_dict, device, dataset_name, model):
    target_model.eval()
    results_list = []
    Du_loader = dataloaders_dict['Du']
    for index in range(len(test_dict['model'])):
        model_name = test_dict['name'][index]
        attack_models = test_dict['model'][index]
        print(model_name)
        base_model[0].eval()
        base_model[1].eval()
        for name, loader in dataloaders_dict.items():
            total_fidelity_kl = 0.0
            total_fidelity_kl_base = 0.0
            total_agreement = 0
            total_base_agreement = 0
            total_consistent = 0
            total_samples = 0
            total_correct_target = 0
            total_correct_ensemble = 0
            target_Rtruth = []
            ensemble_Rtruth = []
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits_target = target_model(inputs)
                    logits_target_agree = base_model[0](inputs)

                    probs_ensemble = ensemble_predictions_prob_average(attack_models, inputs, device)
                    probs_ensemble_agree = ensemble_predictions_prob_average([base_model[1]], inputs, device)

                    logits_ensemble = probs_ensemble.log()
                    logits_ensemble_base = probs_ensemble_agree.log()
                    batch_target_Rtruth = calculate_batch_r_truth(logits_target, labels)
                    batch_ensemble_Rtruth = calculate_batch_r_truth(logits_ensemble, labels)

                    batch_fidelity_kl = compute_kl_fidelity_batch(logits_target, logits_ensemble)
                    batch_fidelity_kl_base = compute_kl_fidelity_batch(logits_target_agree, logits_ensemble_base)
                    batch_agreement = (torch.argmax(logits_target, dim=1) == torch.argmax(probs_ensemble, dim=1)).sum().item()
                    batch_agreement_base = (torch.argmax(logits_target_agree, dim=1) == torch.argmax(probs_ensemble_agree, dim=1)).sum().item()

                    preds_target = torch.argmax(logits_target, dim=1)
                    preds_ensemble = torch.argmax(probs_ensemble, dim=1)
                    batch_consistent = compute_consistency(preds_target, preds_ensemble, labels)

                    total_correct_target += compute_accuracy(logits_target, labels)
                    total_correct_ensemble += (torch.argmax(probs_ensemble, dim=1) == labels).sum().item()

                    target_Rtruth.append(batch_target_Rtruth)
                    ensemble_Rtruth.append(batch_ensemble_Rtruth)
                    total_fidelity_kl += batch_fidelity_kl
                    total_fidelity_kl_base += batch_fidelity_kl_base
                    total_base_agreement += batch_agreement_base
                    total_agreement += batch_agreement
                    total_consistent += batch_consistent
                    total_samples += inputs.size(0)

            target_Rtruth = torch.cat(target_Rtruth, dim=0)
            ensemble_Rtruth = torch.cat(ensemble_Rtruth, dim=0)
            p_value = compute_ks_p_value(normalize_r_truth(target_Rtruth), normalize_r_truth(ensemble_Rtruth))
            avg_fidelity_kl = total_fidelity_kl / total_samples
            avg_fidelity_kl_base = total_fidelity_kl_base / total_samples
            avg_agreement = total_agreement / total_samples
            avg_agreement_base = total_base_agreement / total_samples
            avg_consistent = total_consistent / total_samples
            acc_target = total_correct_target / total_samples
            acc_ensemble = total_correct_ensemble / total_samples
            results_list.append({
                "Dataset": name,
                "Ensemble Accuracy": round(acc_ensemble*100, 2),
                "Model": model_name,
                'Modelname': model,
                "Target Accuracy": round(acc_target*100, 2),
                "Fidelity_KL": round(avg_fidelity_kl, 3),
                "Agreement Diff": round(abs(avg_agreement - avg_agreement_base)*100, 2),
                "Agreement": round(avg_agreement*100, 2),
                "Consistency": round(avg_consistent*100, 2),
                "Base Agreement": round(avg_agreement_base*100, 3),
                "Fidelity_KL_Base": round(avg_fidelity_kl_base, 3),
            })
    df = pd.DataFrame(results_list)
    return df

def evaluate_with_prob_average_adapt(target_model, test_dict, base_model, dataloaders_dict, device):
    target_model.eval()
    Du_loader = dataloaders_dict['Du']
    acc = acc_calculate(target_model, Du_loader, device) * 100
    mv_acc = test_dict['Du']
    index, _ = min(
        enumerate(mv_acc),
        key=lambda x: abs(x[1] - acc)
    )
    model_name = test_dict['name'][index]
    attack_models = test_dict['model'][index]
    base_model[0].eval()
    base_model[1].eval()
    for name, loader in dataloaders_dict.items():
        total_fidelity_kl = 0.0
        total_fidelity_kl_base = 0.0
        total_agreement = 0
        total_base_agreement = 0
        total_consistent = 0
        total_samples = 0
        total_correct_target = 0
        total_correct_ensemble = 0
        target_Rtruth = []
        ensemble_Rtruth = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_target = target_model(inputs)
                logits_target_agree = base_model[0](inputs)

                probs_ensemble = ensemble_predictions_prob_average(attack_models, inputs, device)
                probs_ensemble_agree = ensemble_predictions_prob_average([base_model[1]], inputs, device)

                logits_ensemble = probs_ensemble.log()
                logits_ensemble_base = probs_ensemble_agree.log()

                batch_target_Rtruth = calculate_batch_r_truth(logits_target, labels)
                batch_ensemble_Rtruth = calculate_batch_r_truth(logits_ensemble, labels)

                batch_fidelity_kl = compute_kl_fidelity_batch(logits_target, logits_ensemble)
                batch_fidelity_kl_base = compute_kl_fidelity_batch(logits_target_agree, logits_ensemble_base)
                batch_agreement = (torch.argmax(logits_target, dim=1) == torch.argmax(probs_ensemble, dim=1)).sum().item()
                batch_agreement_base = (torch.argmax(logits_target_agree, dim=1) == torch.argmax(probs_ensemble_agree, dim=1)).sum().item()

                preds_target = torch.argmax(logits_target, dim=1)
                preds_ensemble = torch.argmax(probs_ensemble, dim=1)
                batch_consistent = compute_consistency(preds_target, preds_ensemble, labels)

                total_correct_target += compute_accuracy(logits_target, labels)
                total_correct_ensemble += (torch.argmax(probs_ensemble, dim=1) == labels).sum().item()

                target_Rtruth.append(batch_target_Rtruth)
                ensemble_Rtruth.append(batch_ensemble_Rtruth)
                total_fidelity_kl += batch_fidelity_kl
                total_fidelity_kl_base += batch_fidelity_kl_base
                total_base_agreement += batch_agreement_base
                total_agreement += batch_agreement
                total_consistent += batch_consistent
                total_samples += inputs.size(0)

        target_Rtruth = torch.cat(target_Rtruth, dim=0)
        ensemble_Rtruth = torch.cat(ensemble_Rtruth, dim=0)
        p_value = compute_ks_p_value(normalize_r_truth(target_Rtruth), normalize_r_truth(ensemble_Rtruth))
        avg_fidelity_kl = total_fidelity_kl / total_samples
        avg_fidelity_kl_base = total_fidelity_kl_base / total_samples
        avg_agreement = total_agreement / total_samples
        avg_agreement_base = total_base_agreement / total_samples
        avg_consistent = total_consistent / total_samples
        acc_target = total_correct_target / total_samples
        acc_ensemble = total_correct_ensemble / total_samples

        print(f"Results for {name}:")
        print(f"  Unlearn Model Accuracy: {acc_target * 100:.2f}%")
        print(f"  Small Model Accuracy: {acc_ensemble * 100:.2f}%")
        print(f"  Small Model : {model_name}")
        print(f"  Average Fidelity (KL Divergence): {avg_fidelity_kl}")
        print(f"  Average Fidelity_base (KL Divergence): {avg_fidelity_kl_base}")
        print(f"  Average Agreement Rate diff: {abs(avg_agreement - avg_agreement_base) * 100:.2f}%")
        print(f"  Average Agreement Rate: {avg_agreement * 100:.2f}%")
        print(f"  Average base Agreement Rate: {avg_agreement_base * 100:.2f}%")
        print(f"  Average Consistent Rate: {avg_consistent * 100:.2f}%")


        print("-" * 50)

def evaluate_with_prob_average(target_model, attack_models, base_model, dataloaders_dict, device):
    target_model.eval()
    base_model[0].eval()
    base_model[1].eval()
    for name, loader in dataloaders_dict.items():
        total_fidelity_kl = 0.0
        total_fidelity_kl_base = 0.0
        total_agreement = 0
        total_base_agreement = 0
        total_consistent = 0
        total_samples = 0
        total_correct_target = 0
        total_correct_ensemble = 0
        target_Rtruth = []
        ensemble_Rtruth = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_target = target_model(inputs)
                logits_target_agree = base_model[0](inputs)

                probs_ensemble = ensemble_predictions_prob_average(attack_models, inputs, device)
                probs_ensemble_agree = ensemble_predictions_prob_average([base_model[1]], inputs, device)

                logits_ensemble = probs_ensemble.log()
                logits_ensemble_base = probs_ensemble_agree.log()
                batch_target_Rtruth = calculate_batch_r_truth(logits_target, labels)
                batch_ensemble_Rtruth = calculate_batch_r_truth(logits_ensemble, labels)

                batch_fidelity_kl = compute_kl_fidelity_batch(logits_target, logits_ensemble)
                batch_fidelity_kl_base = compute_kl_fidelity_batch(logits_target_agree, logits_ensemble_base)
                batch_agreement = (
                            torch.argmax(logits_target, dim=1) == torch.argmax(probs_ensemble, dim=1)).sum().item()
                batch_agreement_base = (torch.argmax(logits_target_agree, dim=1) == torch.argmax(probs_ensemble_agree,
                                                                                                 dim=1)).sum().item()

                preds_target = torch.argmax(logits_target, dim=1)
                preds_ensemble = torch.argmax(probs_ensemble, dim=1)
                batch_consistent = compute_consistency(preds_target, preds_ensemble, labels)

                total_correct_target += compute_accuracy(logits_target, labels)
                total_correct_ensemble += (torch.argmax(probs_ensemble, dim=1) == labels).sum().item()

                target_Rtruth.append(batch_target_Rtruth)
                ensemble_Rtruth.append(batch_ensemble_Rtruth)
                total_fidelity_kl += batch_fidelity_kl
                total_fidelity_kl_base += batch_fidelity_kl_base
                total_base_agreement += batch_agreement_base
                total_agreement += batch_agreement
                total_consistent += batch_consistent
                total_samples += inputs.size(0)

        target_Rtruth = torch.cat(target_Rtruth, dim=0)
        ensemble_Rtruth = torch.cat(ensemble_Rtruth, dim=0)
        p_value = compute_ks_p_value(normalize_r_truth(target_Rtruth), normalize_r_truth(ensemble_Rtruth))
        avg_fidelity_kl = total_fidelity_kl / total_samples
        avg_fidelity_kl_base = total_fidelity_kl_base / total_samples
        avg_agreement = total_agreement / total_samples
        avg_agreement_base = total_base_agreement / total_samples
        avg_consistent = total_consistent / total_samples
        acc_target = total_correct_target / total_samples
        acc_ensemble = total_correct_ensemble / total_samples

        print(f"Results for {name}:")
        print(f"  Unlearn Model Accuracy: {acc_target * 100:.2f}%")
        print(f"  Small Model Accuracy: {acc_ensemble * 100:.2f}%")
        print(f"  Average Fidelity (KL Divergence): {avg_fidelity_kl}")
        print(f"  Average Fidelity_base (KL Divergence): {avg_fidelity_kl_base}")
        print(f"  Average Agreement Rate diff: {abs(avg_agreement - avg_agreement_base) * 100:.2f}%")
        print(f"  Average Agreement Rate: {avg_agreement * 100:.2f}%")
        print(f"  Average base Agreement Rate: {avg_agreement_base * 100:.2f}%")
        print(f"  Average Consistent Rate: {avg_consistent * 100:.2f}%")

        print("-" * 50)


def evaluate_with_prob_average_base(target_model, attack_models, dataloaders_dict, device):
    target_model.eval()
    attack_models.eval()
    for name, loader in dataloaders_dict.items():
        total_fidelity_kl = 0.0
        total_agreement = 0
        total_consistent = 0
        total_samples = 0
        total_correct_target = 0
        total_correct_ensemble = 0
        target_Rtruth = []
        ensemble_Rtruth = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits_target = target_model(inputs)

                probs_ensemble = F.softmax(attack_models(inputs.to(device)), dim=1)

                logits_ensemble = probs_ensemble.log()
                batch_target_Rtruth = calculate_batch_r_truth(logits_target, labels)
                batch_ensemble_Rtruth = calculate_batch_r_truth(logits_ensemble, labels)

                batch_fidelity_kl = compute_kl_fidelity_batch(logits_target, logits_ensemble)
                batch_agreement = (
                            torch.argmax(logits_target, dim=1) == torch.argmax(probs_ensemble, dim=1)).sum().item()

                preds_target = torch.argmax(logits_target, dim=1)
                preds_ensemble = torch.argmax(probs_ensemble, dim=1)
                batch_consistent = compute_consistency(preds_target, preds_ensemble, labels)

                total_correct_target += compute_accuracy(logits_target, labels)
                total_correct_ensemble += (torch.argmax(probs_ensemble, dim=1) == labels).sum().item()

                target_Rtruth.append(batch_target_Rtruth)
                ensemble_Rtruth.append(batch_ensemble_Rtruth)
                total_fidelity_kl += batch_fidelity_kl
                total_agreement += batch_agreement
                total_consistent += batch_consistent
                total_samples += inputs.size(0)

        target_Rtruth = torch.cat(target_Rtruth, dim=0)
        ensemble_Rtruth = torch.cat(ensemble_Rtruth, dim=0)
        p_value = compute_ks_p_value(normalize_r_truth(target_Rtruth), normalize_r_truth(ensemble_Rtruth))
        avg_fidelity_kl = total_fidelity_kl / total_samples
        avg_agreement = total_agreement / total_samples
        avg_consistent = total_consistent / total_samples
        acc_target = total_correct_target / total_samples
        acc_ensemble = total_correct_ensemble / total_samples

        print(f"Results for {name}:")
        print(f"  Unlearn Model Accuracy: {acc_target * 100:.2f}%")
        print(f"  Small Model Accuracy: {acc_ensemble * 100:.2f}%")
        print(f"  Average Fidelity (KL Divergence): {avg_fidelity_kl}")
        print(f"  Average Agreement Rate: {avg_agreement * 100:.2f}%")
        print(f"  Average Consistent Rate: {avg_consistent * 100:.2f}%")


        print("-" * 50)


def ensemble_predictions_majority_vote(attack_models, inputs, device):
    preds_attacks = torch.stack(
        [torch.argmax(model(inputs.to(device)), dim=1) for model in attack_models],
        dim=0
    )

    preds_ensemble, _ = torch.mode(preds_attacks, dim=0)

    return preds_ensemble


def evaluate_with_majority_vote(target_model, attack_models, dataloaders_dict, device):
    target_model.eval()
    for model in attack_models:
        model.eval()

    for name, loader in dataloaders_dict.items():
        total_agreement = 0
        total_samples = 0
        total_correct_target = 0
        total_correct_ensemble = 0
        total_consistency = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                logits_target = target_model(inputs)
                preds_target = torch.argmax(logits_target, dim=1)

                preds_ensemble = ensemble_predictions_majority_vote(attack_models, inputs, device)

                total_correct_target += (preds_target == labels).sum().item()
                total_correct_ensemble += (preds_ensemble == labels).sum().item()
                total_agreement += (preds_target == preds_ensemble).sum().item()

                consistent = compute_consistency(preds_target, preds_ensemble, labels)
                total_consistency += consistent

                total_samples += inputs.size(0)

        acc_target = total_correct_target / total_samples
        acc_ensemble = total_correct_ensemble / total_samples
        avg_agreement = total_agreement / total_samples
        avg_consistency = total_consistency / total_samples

        # 打印每个 DataLoader 的评估结果
        print(f"Results for {name}:")
        print(f"  Retrained Model Accuracy: {acc_target * 100:.2f}%")
        print(f"  Ensemble Model Accuracy: {acc_ensemble * 100:.2f}%")
        print(f"  Average Agreement Rate: {avg_agreement * 100:.2f}%")
        print(f"  Consistency Rate: {avg_consistency * 100:.2f}%")
        print("-" * 50)


