from torchvision import datasets, transforms
import torch as ch
from torch.utils.data.dataset import Subset
BATCH_SIZE = 64

import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
])

fullset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

trainloaderfull = ch.utils.data.DataLoader(fullset, batch_size=BATCH_SIZE, shuffle=True)
MNIST_testloader = ch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

trainset = Subset(fullset,range(55000))
valset = Subset(fullset, range(55000,60000))
MNIST_trainloader = ch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
MNIST_validationloader = ch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

fashionmnist = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_fashionmnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

x_fashionmnist,y_fashionmnist = [],[]
trn_x_fashionmnist,trn_y_fashionmnist = [],[]
for i in range(len(fashionmnist)):
    x_fashionmnist.append(fashionmnist[i][0])
    y_fashionmnist.append(fashionmnist[i][1])
for i in range(len(train_fashionmnist)):
    trn_x_fashionmnist.append(train_fashionmnist[i][0])
    trn_y_fashionmnist.append(train_fashionmnist[i][1])

fashionmnist = torch.utils.data.TensorDataset(torch.stack(x_fashionmnist), torch.tensor(y_fashionmnist))
train_fashionmnist = torch.utils.data.TensorDataset(torch.stack(trn_x_fashionmnist), torch.tensor(trn_y_fashionmnist))
FM_trainset = Subset(train_fashionmnist, range(55000))
FM_valset = Subset(train_fashionmnist, range(55000,60000))
FM_train_loader = ch.utils.data.DataLoader(FM_trainset, batch_size=BATCH_SIZE, shuffle=True)
FM_val_loader = ch.utils.data.DataLoader(FM_valset, batch_size=BATCH_SIZE, shuffle=True)
FM_test_loader = ch.utils.data.DataLoader(fashionmnist, batch_size=BATCH_SIZE, shuffle=False)

cifar_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

cifar_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

cifar_train_set = datasets.CIFAR10(root=".data", train=True, transform=cifar_transform, download=True)
cifar_test_set = datasets.CIFAR10(root=".data", train=False, transform=cifar_test_transform, download=True)
cifar_trainset = Subset(cifar_train_set, range(40000))
cifar_valset = Subset(cifar_train_set, range(40000,50000))
cifar_train_loader = ch.utils.data.DataLoader(cifar_trainset, batch_size=BATCH_SIZE, shuffle=True)
cifar_val_loader = ch.utils.data.DataLoader(cifar_valset, batch_size=BATCH_SIZE, shuffle=True)
cifar_test_loader = ch.utils.data.DataLoader(cifar_test_set, batch_size=BATCH_SIZE, shuffle=False)

train_svhn = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
svhn_trainset = Subset(train_svhn, range(50000))
svhn_valset = Subset(train_svhn, range(50000,73257))
svhn_testset = datasets.SVHN(root='./data', split='test', transform=transform, download=True)
svhn_train_loader = ch.utils.data.DataLoader(svhn_trainset, batch_size=BATCH_SIZE, shuffle=True)
svhn_val_loader = ch.utils.data.DataLoader(svhn_valset, batch_size=BATCH_SIZE, shuffle=True)
svhn_test_loader = ch.utils.data.DataLoader(svhn_testset, batch_size=BATCH_SIZE, shuffle=False)



def get_tiny_image_net(batch_size=64, seed=1):

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = './tiny-imagenet-200/train'

    trainset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    test_dir = './tiny-imagenet-200/val'
    testset = torchvision.datasets.ImageFolder(
        test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)


    return trainset, testset, train_loader, test_loader

def get_cifar10_resized(batch_size=64, seed=1, dim=64):

    transform_train = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return trainset, testset, train_loader, test_loader

def get_svhn_resized(batch_size=64, seed=1, dim=64):

    transform_train = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                          download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test',
                                         download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return trainset, testset, train_loader, test_loader

def get_tiny32(batch_size=64, seed=1, dim=32):

    transform_test = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dir = './tiny-imagenet-200/val'
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)


    return testset

from torchvision.datasets import CIFAR10, SVHN, CIFAR100

def get_cifar10_28():
    """
    Returns the CIFAR-10 test set converted to grayscale and resized to FashionMNIST image dimensions (28x28).

    Returns:
        torch.utils.data.Dataset: CIFAR-10 test set with transformed images.
    """
    # Define the transformations: Convert to grayscale and resize to 28x28
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                # Resize to FashionMNIST dimensions
        transforms.ToTensor(),                       # Convert to tensor
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load CIFAR-10 test set with the transformations
    cifar10_test_set = CIFAR10(root="./data", train=False, download=True, transform=transform)

    return cifar10_test_set

def get_svhn_28():
    """
    Returns the SVHN test set converted to grayscale and resized to FashionMNIST image dimensions (28x28).

    Returns:
        torch.utils.data.Dataset: SVHN test set with transformed images.
    """
    # Define the transformations: Convert to grayscale and resize to 28x28
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                # Resize to FashionMNIST dimensions
        transforms.ToTensor(),                       # Convert to tensor
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load SVHN test set with the transformations
    svhn_test_set = SVHN(root="./data", split="test", download=True, transform=transform)

    return svhn_test_set

def get_cifar100_32():
    """
    Returns the CIFAR100 test set converted to grayscale and resized to FashionMNIST image dimensions (28x28).

    Returns:
        torch.utils.data.Dataset: CIFAR100 test set with transformed images.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR100 test set with the transformations
    cifar100_test_set = CIFAR100(root="./data", train=False, download=True, transform=transform)

    return cifar100_test_set

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_human(data_dir, batch_size=64, dim=64):
    """
    Create train_loader, test_loader, and test_set for human detection dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for data loaders.

    Returns:
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing.
        test_set: The test dataset.
    """
    # Transformations for training and testing
    transform_train = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(dim, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

    # Calculate split sizes
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size

    # Split the dataset
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    # Apply test transformations to the test set
    test_set.dataset.transform = transform_test

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, test_set


import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
