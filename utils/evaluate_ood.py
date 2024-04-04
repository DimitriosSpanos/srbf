import ffmpeg.nodes
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from utils.datasets import (
    get_CIFAR10,
    get_SVHN,
    get_FashionMNIST,
    get_MNIST,
    get_notMNIST,
)

train_device = "cuda:2"

def prepare_ood_datasets(true_dataset, ood_dataset):
    # Preprocess OoD dataset same as true dataset
    #ood_dataset.transform = true_dataset.transform
    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, dataloader, standard_model):
    model.eval()
    global train_device
    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            data = data.to(train_device)
            target = target.to(train_device)

            if standard_model:
                output = model(data)
                _, pred = output.max(1)
                probs = F.softmax(output, dim=1)
                uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            else:
                output, _, _, _ = model(data)
                kernel_distance, pred = output.max(1)
                uncertainty = - kernel_distance
            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores.append(uncertainty.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

from matplotlib import pyplot as plt

def get_auroc_ood(true_dataset, ood_dataset, model, device, standard_model=False, final=False):
    global train_device
    train_device = device
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader, standard_model)

    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)

    if final:
        uncertainties_class_0 = [-uncertainty for uncertainty, label in zip(scores, anomaly_targets) if label == 0]
        uncertainties_class_1 = [-uncertainty for uncertainty, label in zip(scores, anomaly_targets) if label == 1]

        # Plot histogram with different colors for each class
        plt.hist([uncertainties_class_0, uncertainties_class_1], bins=10, color=['blue', 'orange'],
                 label=['In Domain', 'Out of Distribution'])
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('CIFAR-10 Proposed Confidence Score Distribution')
        plt.ylim(0, 10000)
        plt.legend()
        plt.savefig('Proposed_histogram_CIFAR-10.pdf', format='pdf')
        plt.show()

    return accuracy, roc_auc


def get_auroc_classification(dataset, model):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc


def get_cifar_svhn_ood(model):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()

    return get_auroc_ood(cifar_test_dataset, svhn_test_dataset, model)


def get_fashionmnist_mnist_ood(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()

    return get_auroc_ood(fashionmnist_test_dataset, mnist_test_dataset, model)


def get_fashionmnist_notmnist_ood(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, notmnist_test_dataset = get_notMNIST()

    return get_auroc_ood(fashionmnist_test_dataset, notmnist_test_dataset, model)
