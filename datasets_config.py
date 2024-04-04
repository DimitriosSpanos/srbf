from torchvision import datasets, transforms
import torch as ch
from torch.utils.data.dataset import Subset
BATCH_SIZE = 128

import torch
import torchvision
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
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



def get_tiny_image_net(batch_size=128, seed=1):

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

def get_cifar10_resized(batch_size=128, seed=1):

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
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

def get_svhn_resized(batch_size=128, seed=1):

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
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
