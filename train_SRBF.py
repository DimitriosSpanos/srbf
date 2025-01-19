import torch
import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys, pylab, random, time

from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from utils.progress import progress_bar
from utils.my_utils import proximity_loss, plot_centers_and_features, center_proximity_loss,get_feature_centers
from utils.my_utils import subclass_identification
from models.SRBF import SRBF
from utils.evaluate_ood import get_auroc_ood
from models.autoencoders import CAE, CAE2, CAE3, CAE5
import matplotlib.pyplot as plt
from utils.gradient_penalty import calc_gradient_penalty
from datasets_config import get_tiny_image_net, get_cifar10_resized, get_svhn_resized, get_human

parser = ArgumentParser()
parser.add_argument('--dataset', choices=["mnist", "cifar", "gaussian", "svhn", "FM", "tiny_imagenet", "human"], required=True,
        help="Which dataset to use for training")
parser.add_argument('--opt', default="sgd", choices=["adam", "sgd"])
parser.add_argument('--sgd-lr', type=float, default=1e-2,
        help="SGD learning rate")
parser.add_argument('--sigma', type=float, default=10,
        help="length scale of RBF kernel")
parser.add_argument('--gamma', type=float, default=0.999,
        help="decay factor for exponential average")
parser.add_argument('--save-str', type=str, default="",
        help="A unique identifier to save with")
parser.add_argument('--resume', type=str)
parser.add_argument('--num-epochs', default=500, type=int,
        help="Number of epochs to train for")
parser.add_argument('--use_gp', action='store_true',
        help="Whether to use Gradient Penalty / Double Propagation")
parser.add_argument('--AE-pretrained', action='store_true',
        help="Whether to use pretrained AE")
parser.add_argument('--subclass', type=int, default=0,
        help='number of epoch to start subclass learning, if 0 then doesnt use')
parser.add_argument('--cs', default=128, type=int,
        help="centroid size")
parser.add_argument('--train-device', type=str, default='cpu',
                            help="Device to train the network. Use 'cuda' for the GPU."
                                 "Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")

args = parser.parse_args()
train_device = args.train_device


# Function to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.dataset == "cifar" or args.dataset == "svhn":
    from datasets_config import cifar_trainset, cifar_train_loader, cifar_test_loader, cifar_test_set, cifar_val_loader, cifar_valset
    from datasets_config import train_svhn,svhn_testset, svhn_train_loader, svhn_val_loader, svhn_test_loader, svhn_valset, svhn_trainset
    if args.dataset == "cifar":
        trainloader, validationloader, testloader = cifar_train_loader, cifar_val_loader, cifar_test_loader
        val_auroc1, val_auroc2 = cifar_test_set, svhn_testset
        in_test, out_test = cifar_test_set, svhn_testset
        ood_val_loader = svhn_val_loader
    elif args.dataset == "svhn":
        trainloader, validationloader, testloader = svhn_train_loader, svhn_val_loader, svhn_test_loader
        val_auroc1, val_auroc2 = svhn_testset, cifar_test_set
        in_test, out_test = svhn_testset, cifar_test_set
        ood_val_loader = cifar_val_loader
elif args.dataset == "tiny_imagenet":
    tiny_train, tiny_test, tiny_train_loader, tiny_test_loader = get_tiny_image_net()
    cifar_train, cifar_test, cifar_train_loader, cifar_test_loader = get_cifar10_resized()
    svhn_train, svhn_test, svhn_train_loader, svhn_test_loader = get_svhn_resized()
    trainloader, validationloader, testloader = tiny_train_loader, tiny_train_loader, tiny_test_loader
    val_auroc1, val_auroc2 = tiny_test, cifar_test
    in_test, out_test = tiny_test, cifar_test
    ood_val_loader = cifar_test_loader
    second_ood = svhn_test
elif args.dataset == "human":

    _, tiny_test, _, tiny_test_loader = get_tiny_image_net()
    _, cifar_test, _, _ = get_cifar10_resized()
    _, svhn_test, _, _ = get_svhn_resized()
    trainloader, validationloader, test_set = get_human("./human_detection_dataset")
    testloader= validationloader
    val_auroc1, val_auroc2 = test_set, tiny_test
    in_test, out_test = test_set, tiny_test
    ood_val_loader = tiny_test_loader
    second_ood = svhn_test
    third_ood = cifar_test
else:
    from datasets_config import fashionmnist, FM_train_loader, FM_val_loader, FM_test_loader, FM_trainset, FM_valset
    from datasets_config import MNIST_trainloader, MNIST_validationloader, MNIST_testloader, testset, valset
    if args.dataset == "mnist":
        ood_val_loader = FM_val_loader
        trainloader, validationloader, testloader = MNIST_trainloader, MNIST_validationloader, MNIST_testloader
        val_auroc1, val_auroc2 = valset, FM_valset
        in_test, out_test = testset, fashionmnist
    elif args.dataset == "FM":
        ood_val_loader = MNIST_validationloader
        trainloader, validationloader, testloader = FM_train_loader, FM_val_loader, FM_test_loader
        val_auroc1, val_auroc2 = FM_valset, valset
        in_test, out_test = fashionmnist, testset

num_classes = 2
if args.dataset == "cifar":
    model_output_size = 512
    binary_map = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  # vehicle:0, animal:1
    args.sigma = 1.
elif args.dataset == "svhn":
    model_output_size = 512
    binary_map = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    if not args.subclass:
        if args.AE_pretrained:
            args.sigma = 1.4
        else:
            args.sigma = 0.07
elif args.dataset == "tiny_imagenet":
    model_output_size = 512
    binary_map = [0] * 100 + [1] * 100
    if not args.subclass:
        if args.AE_pretrained:
            args.sigma = 4.6
        else:
            args.sigma = 0.14
elif args.dataset == "human":
    model_output_size = 512
    binary_map = [0, 1]
    if not args.subclass:
        if args.AE_pretrained:
            args.sigma = 4.6
        else:
            args.sigma = 0.2
elif args.dataset == "mnist":
    model_output_size = 64
    binary_map = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    if not args.subclass:
        if args.AE_pretrained:
            args.sigma = 1.5
        else:
            args.sigma = 0.04
elif args.dataset == "FM":
    model_output_size = 256
    binary_map = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    if not args.subclass:
        if args.AE_pretrained:
            args.sigma = 7.5
        else:
            args.sigma = 0.1
def run_training(seed):
    set_seed(seed)
    net = SRBF(None,
                  num_classes,
                  args.cs,
                  model_output_size,
                  float(args.sigma),   # help="Length scale of RBF kernel",
                  args.gamma, # help="Decay factor for exponential average (default: 0.999)"
                  train_device,
                  args.dataset,
                  binary_map
                 )
    if args.resume:
        net_dict = ch.load("results/%s_%s_%s" % (args.dataset, MODE, args.resume))
        net.load_state_dict(net_dict)

    if args.dataset == "mnist":
        autoencoder = CAE(64)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_mnist.pth", map_location="cpu")
        l_gradient_penalty = 0.05
    elif args.dataset == "FM":
        autoencoder = CAE(256)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_fm.pth", map_location="cpu")
        l_gradient_penalty = 0.05
    elif args.dataset == "cifar":
        autoencoder = CAE2(512)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_cifar10.pth", map_location="cpu")
        l_gradient_penalty = 0.5
    elif args.dataset == "svhn":
        autoencoder = CAE2(512)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_svhn.pth", map_location="cpu")
        l_gradient_penalty = 0.5
    elif args.dataset == "tiny_imagenet":
        autoencoder = CAE3(512)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_tiny_imagenet.pth", map_location="cpu")
        l_gradient_penalty = 0.01
    elif args.dataset == "human":
        autoencoder = CAE5(512)
        if args.AE_pretrained:
            state_dict = ch.load("autoencoder_human18.pth", map_location="cpu")
        l_gradient_penalty = 0.0001
    if args.AE_pretrained:
        autoencoder.load_state_dict(state_dict)
    net.feature_extractor = autoencoder.encoder
    net = net.cuda()
    loss_fn = ch.nn.BCELoss()
    param_set = net.parameters()

    if args.dataset == 'cifar':
        baseline_learn_sigma = 0.
    else:
        baseline_learn_sigma = 1.
    if args.opt == "sgd":
        opt = optim.SGD([
                    {'params': net.feature_extractor.parameters()},
                    {'params': net.W1_part1},
                    {'params': net.W1_part2},
                    {'params': net.sigma1, 'lr': baseline_learn_sigma*args.sgd_lr/100},
                ], lr=args.sgd_lr, momentum=0.9, weight_decay=1e-5)

        if args.dataset in ["cifar","svhn","tiny_imagenet", "human"]:
            scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[60], gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[25], gamma=0.1)

    train_losses, val_losses, train_accs, val_accs, all_aurocs = [], [], [], [], []


    def get_AUROC(model, submnist1, submnist2, final=False):
        return get_auroc_ood(submnist1, submnist2, model, train_device, final=final)


    for ep in range(1,args.num_epochs+1):
        # print(net.mask)
        total_ims_seen, val_num_correct, val_num_total, num_correct, num_total, training_batches = 0,0,0,0,0,0
        training_running_class_loss, training_running_subclass_loss = 0., 0.

        # print(net.sigma1)
        # Subclass identification
        if args.subclass == ep:
            subclass_identification(trainloader, net, ep, binary_map, train_device)
            #plot_centers_and_features(net, validationloader, binary_map, train_device, ood_val_loader, ep>=args.subclass)
            if args.dataset == "cifar":
                learn_sigma = 0.
            else:
                learn_sigma = 1.
            new_params = [{'params': net.W_s, 'lr': args.sgd_lr},
                          {'params': net.sigma1, 'lr': learn_sigma*args.sgd_lr/100},
                          {'params': net.m1, 'lr': args.sgd_lr/100}]
            for param in new_params:
                opt.add_param_group(param)

        for i, (images, labels) in enumerate(trainloader):

            net.train()
            binary_labels = ch.tensor([binary_map[i] for i in labels])
            images, labels, binary_labels = images.cuda(), labels.cuda(), binary_labels.cuda()
            opt.zero_grad()
            images.requires_grad_(True)
            pred_probs, subclass_pred_probs, features, _ = net(images)
            target_one_hot = torch.zeros(pred_probs.size()).cuda()
            target_one_hot.scatter_(1, binary_labels.unsqueeze(1), 1)
            class_loss = loss_fn(pred_probs.float(), target_one_hot)
            pred_classes = pred_probs.argmax(1)
            num_correct += (pred_classes == binary_labels).float().sum()
            num_total += binary_labels.shape[0]
            train_acc = 100. * (num_correct / num_total)
            training_running_class_loss += class_loss.clone().cpu().item()
            training_batches += 1

            if args.use_gp:
                class_loss += l_gradient_penalty * calc_gradient_penalty(images, pred_probs.sum(1))

            # progress_bar(i, len(trainloader),
            #              f'Epoch: {ep} | Class Loss: {training_running_class_loss / training_batches:.4f} |'
            #              f' Train Acc: {train_acc:.4f}')
            images.requires_grad_(False)
            with ch.no_grad():
                if ep >= args.subclass and args.subclass:
                    y = subclass_pred_probs.float()
                else:
                    y = F.one_hot(binary_labels, num_classes).float()
                net.eval()

                net.update_embeddings(images, y, ep>=args.subclass)
                net.train()
            total_ims_seen += images.shape[0]
            class_loss.backward()
            opt.step()

        net.eval()
        validation_running_class_loss, validation_running_subclass_loss, validation_batches = 0., 0., 0
        for i, (images, labels) in enumerate(validationloader):
            net.eval()
            binary_labels = ch.tensor([binary_map[j] for j in labels])
            images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(train_device)
            pred_probs, pred_subclass_probs, _, _ = net(images)
            y = F.one_hot(binary_labels, num_classes).float()
            target_one_hot = torch.zeros(pred_probs.size()).to(train_device)
            target_one_hot.scatter_(1, binary_labels.unsqueeze(1), 1)
            class_loss = loss_fn(pred_probs.float(), target_one_hot)
            pred_classes = pred_probs.argmax(1)
            val_num_correct += (pred_classes == binary_labels).float().sum()
            val_num_total += binary_labels.shape[0]
            validation_running_class_loss += class_loss.clone().cpu().item()
            validation_batches += 1
            val_acc = 100. * (val_num_correct / val_num_total)
        print(f"Epoch: {ep} | Validation Accuracy: {val_acc}")

        scheduler.step()
        train_losses.append(training_running_class_loss / training_batches)
        val_losses.append(validation_running_class_loss / validation_batches)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        #ch.save(net.state_dict(), "results/%s_gpu:%s" % (args.dataset, train_device))

        auroc, aupr = get_auroc_ood(true_dataset=in_test, ood_dataset=out_test, model=net, device="cuda:0",
                                    standard_model=False)
        # print("Auroc val: ", auroc)
        all_aurocs.append(auroc)

    #net.load_state_dict(ch.load("results/%s_gpu:%s" % (args.dataset, train_device)))
    net.eval()
    num_correct, num_total = 0, 0
    for (images, labels) in testloader:
        binary_labels = ch.tensor([binary_map[i] for i in labels])
        images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(train_device)
        pred_probs, subclass_pred_probs, _, _ = net(images)
        pred_classes = pred_probs.argmax(1)
        num_correct += (pred_classes == binary_labels).float().sum()
        num_total += binary_labels.shape[0]

    auroc, aupr = get_auroc_ood(true_dataset=in_test, ood_dataset=out_test, model=net, device="cuda:0",
                                standard_model=False)
    print("-------------------------")
    print("Test Accuracy: %f" % (num_correct / num_total * 100).cpu().item())
    print(f"Test Auroc: {auroc:.4f}")
    print(f"Last 5 AUROC: {(sum(all_aurocs[-5:]) / len(all_aurocs[-5:])):.4f}")
    print("-------------------------")

    if args.dataset == "tiny_imagenet" or args.dataset == "human":
        auroc2, aupr = get_auroc_ood(true_dataset=in_test, ood_dataset=second_ood, model=net, device="cuda:0",
                                     standard_model=False)
        print(f"Test Auroc: {auroc2:.4f}")
        auroc3=0.
    if args.dataset == "human":
        auroc3, aupr = get_auroc_ood(true_dataset=in_test, ood_dataset=third_ood, model=net, device="cuda:0",
                                     standard_model=False)
        print(f"Test Auroc: {auroc3:.4f}")
    if args.dataset not in ["human", "tiny_imagenet"]:
        auroc2 = 0.
        auroc3 = 0.
    saved_name = f"{args.dataset}.pth"
    torch.save(net.state_dict(), saved_name)
    net = torch.load(saved_name)

    return (num_correct / num_total * 100).cpu().item(), auroc, auroc2, auroc3

# Running for 5 different seeds
results_test_acc = []
results_auroc = []
results_auroc2 = []
results_auroc3 = []

seeds = [random.randint(0,100) for _ in range(5)]
for seed in seeds:
    print(f"Running with seed {seed}")
    acc, auroc, auroc2, auroc3 = run_training(seed)
    results_test_acc.append(acc)
    results_auroc.append(auroc)
    results_auroc2.append(auroc2)
    results_auroc3.append(auroc3)

# Print mean and std for all metrics
print(f"Mean Test Accuracy: {np.mean(results_test_acc):.4f} | Std Test Accuracy: {np.std(results_test_acc):.4f}")
print(f"Mean AUROC: {np.mean(results_auroc):.4f} | Std AUROC: {np.std(results_auroc):.4f}")
print(f"Mean AUROC2: {np.mean(results_auroc2):.4f} | Std AUROC: {np.std(results_auroc2):.4f}")
print(f"Mean AUROC3: {np.mean(results_auroc3):.4f} | Std AUROC: {np.std(results_auroc3):.4f}")