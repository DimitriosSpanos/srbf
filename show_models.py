import torch
import torch as ch
from utils.my_utils import subclass_identification
from models.SRBF import SRBF
from utils.evaluate_ood import get_auroc_ood
from models.autoencoders import CAE, CAE2, CAE3
from datasets_config import get_tiny_image_net, get_cifar10_resized, get_svhn_resized

train_device = "cuda:0"
def get_AUROC(model, in_domain, out_of_domain, final=False):
    return get_auroc_ood(in_domain, out_of_domain, model, train_device, final=final)

def test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, second_ood):
    net.eval()
    num_correct, num_total = 0, 0
    for (images, labels) in testloader:
        binary_labels = ch.tensor([binary_map[i] for i in labels])
        images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(train_device)
        pred_probs, subclass_pred_probs, _, _ = net(images)
        pred_classes = pred_probs.argmax(1)
        num_correct += (pred_classes == binary_labels).float().sum()
        num_total += binary_labels.shape[0]

    accuracy, auroc = get_AUROC(net, in_domain, out_of_domain, final=False)
    print("-------------------------")
    print("Test Accuracy: %f" % (num_correct / num_total * 100).cpu().item())
    print(f"Test Auroc: {auroc:.4f}")
    if dataset == "tiny_imagenet":
        accuracy, second_auroc = get_AUROC(net, in_domain, second_ood, final=False)
        print(f"Test Auroc 2: {second_auroc:.4f}")
    print("-------------------------")


# MNIST
dataset = "mnist"
from datasets_config import MNIST_testloader as testloader
from datasets_config import testset as in_domain
from datasets_config import fashionmnist as out_of_domain
from datasets_config import MNIST_trainloader as trainloader


print(f"\n\n{dataset} evaluation")
num_classes, model_size, centroid_size = 2, 64, 64
binary_map = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
AE = CAE(model_size)
net = SRBF(AE.encoder,num_classes,centroid_size,model_size,1.,0.999,train_device,dataset,binary_map).to(train_device)
subclass_identification(trainloader, net, 1, binary_map, train_device)

net.load_state_dict(torch.load(f"{dataset}.pth"))
test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, out_of_domain)


# Fashion-MNIST
dataset = "FM"
from datasets_config import FM_test_loader as testloader
from datasets_config import fashionmnist as in_domain
from datasets_config import testset as out_of_domain
from datasets_config import FM_train_loader as trainloader

print(f"{dataset} evaluation")
num_classes, model_size, centroid_size = 2, 256, 64
binary_map = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
AE = CAE(model_size)
net = SRBF(AE.encoder,num_classes,centroid_size,model_size,1.,0.999,train_device,dataset,binary_map).to(train_device)
subclass_identification(trainloader, net, 1, binary_map, train_device)

net.load_state_dict(torch.load(f"{dataset}.pth"))
test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, out_of_domain)


# CIFAR-10
dataset = "cifar"
from datasets_config import cifar_test_loader as testloader
from datasets_config import cifar_test_set as in_domain
from datasets_config import svhn_testset as out_of_domain
from datasets_config import svhn_train_loader as trainloader

print(f"{dataset} evaluation")
num_classes, model_size, centroid_size = 2, 512, 512
binary_map = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
AE = CAE2(model_size)
net = SRBF(AE.encoder,num_classes,centroid_size,model_size,1.,0.999,train_device,dataset,binary_map).to(train_device)
subclass_identification(trainloader, net, 1, binary_map, train_device)

net.load_state_dict(torch.load(f"{dataset}.pth"))
test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, out_of_domain)


# SVHN
dataset = "svhn"
from datasets_config import svhn_test_loader as testloader
from datasets_config import svhn_testset as in_domain
from datasets_config import cifar_test_set as out_of_domain
from datasets_config import svhn_train_loader as trainloader

print(f"{dataset} evaluation")
num_classes, model_size, centroid_size = 2, 512, 512
binary_map = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
AE = CAE2(model_size)
net = SRBF(AE.encoder,num_classes,centroid_size,model_size,1.,0.999,train_device,dataset,binary_map).to(train_device)
subclass_identification(trainloader, net, 1, binary_map, train_device)

net.load_state_dict(torch.load(f"{dataset}.pth"))
test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, out_of_domain)


# Tiny ImageNet
dataset = "tiny_imagenet"
_, tiny_test, trainloader, tiny_test_loader = get_tiny_image_net()
_, cifar_test, _, _ = get_cifar10_resized()
_, svhn_test, _, _ = get_svhn_resized()

print(f"{dataset} evaluation")
num_classes, model_size, centroid_size = 2, 512, 512
binary_map = [0] * 100 + [1] * 100
AE = CAE3(model_size)
net = SRBF(AE.encoder,num_classes,centroid_size,model_size,1.,0.999,train_device,dataset,binary_map).to(train_device)
subclass_identification(trainloader, net, 1, binary_map, train_device)

testloader, in_domain, out_of_domain, second_ood = tiny_test_loader, tiny_test, cifar_test, svhn_test
net.load_state_dict(torch.load(f"{dataset}.pth"))

test_model(net, testloader, binary_map, dataset, in_domain, out_of_domain, second_ood)