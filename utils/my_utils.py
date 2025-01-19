import random

from torchvision.transforms.transforms import Lambda
import numpy as np
from PIL import Image
from skimage.feature import hog
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def subclass_identification(trainloader, net, ep, binary_map, train_device, batches_to_use = 100):

    for i, (images, labels) in enumerate(trainloader):
        if i == batches_to_use:
            break
        binary_labels = torch.tensor([binary_map[j] for j in labels])
        images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(train_device)
        _, _, features, _ = net(images)
        if i == 0:
            x, y, s_y = features.cpu().detach(), binary_labels.cpu().detach(), labels.cpu().detach()
        else:
            x, y, s_y = torch.cat((x, features.cpu().detach()), 0), torch.cat((y, binary_labels.cpu().detach()), 0), torch.cat((s_y, labels.cpu().detach()), 0)
    net.criterion(x, y, s_y)
    return

def proximity_loss(centers, features, labels, classes, map):

    proximity = 0
    for class_id in range(classes):
        center_id = map[class_id]
        idx = (labels == class_id).nonzero(as_tuple=True)[0]
        curr_features = features[idx, :, center_id]
        proximity += torch.sum(torch.sum((centers[center_id, :].view(1, -1)-curr_features)**2, dim=1))
    return proximity


def get_feature_centers(features, labels, classes=range(10)):
    curr_centers0, curr_centers1 = [],[]
    for class_id, center_id in enumerate(classes):
        idx = (labels == class_id).nonzero(as_tuple=True)[0]
        curr_features = features[idx, :, center_id]
        if class_id % 2 ==0:
            curr_centers0.append(torch.mean(curr_features, dim=0).cpu().detach().numpy())
        else:
            curr_centers1.append(torch.mean(curr_features, dim=0).cpu().detach().numpy())
    return torch.tensor(np.array(curr_centers0+curr_centers1))

def pairwise_distances(a, b=None, eps=1e-4):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1)
    bb = torch.sum(b ** 2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists


def center_proximity_loss(centers):
    return -torch.mean(pairwise_distances(centers))



# Create the animation
#animation_obj = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=True)

def get_centers(features, labels, classes=range(10)):
    curr_centers = []
    for class_id in classes:
        curr_subclass_idx = (labels == class_id).nonzero(as_tuple=True)[0]
        subclass_items = torch.flatten(torch.index_select(features, 0, curr_subclass_idx), start_dim=1)
        curr_centers.append(torch.mean(subclass_items, dim=0).cpu().detach().numpy())
    return curr_centers



def plot_centers_and_features(net, validationloader, binary_map, train_device, ood_val_loader, subclass_learning):
    features = 0
    svhn_features = 0
    id_labels = 0
    id_sub_labels = 0
    #  ------------ Visualization--
    for i, (images, labels) in enumerate(validationloader):
        net.eval()
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        binary_labels = torch.tensor([binary_map[j] for j in labels])
        images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(
            train_device)
        new_ims = images
        pred_probs, subclass_pred, smaller_features, features = net(new_ims)
        id_labels = binary_labels
        id_sub_labels = labels
        break
    for i, (images, labels) in enumerate(ood_val_loader):
        net.eval()
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        binary_labels = torch.tensor([binary_map[j] for j in labels])
        images, labels, binary_labels = images.to(train_device), labels.to(train_device), binary_labels.to(
            train_device)
        new_ims = images
        svhn_pred_probs, subclass_pred, svhn_smaller_features, svhn_features = net(new_ims)
        od_labels = binary_labels
        break

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, random_state=42)

    if subclass_learning:
        class0_id_idx = (id_labels == 0).nonzero(as_tuple=True)[0].cpu().detach().numpy()
        features0 = features[class0_id_idx, :]
        class1_id_idx = (id_labels == 1).nonzero(as_tuple=True)[0].cpu().detach().numpy()
        features1 = features[class1_id_idx, :]
        svhn_features = svhn_features.cpu().detach().numpy()
        centers = np.array(get_centers(features, id_sub_labels))
        points = np.concatenate(
            (centers, features0.cpu().detach().numpy(), features1.cpu().detach().numpy(), svhn_features), axis=0)
        points = tsne.fit_transform(points)
        plt.scatter(points[10:features0.size(0) + 10, 0], points[10:features0.size(0) + 10, 1], color='blue',
                    marker='o')
        plt.scatter(points[features0.size(0) + 10:138, 0], points[features0.size(0) + 10:138, 1], color='blue',
                    marker='^')
        plt.scatter(points[138:, 0], points[138:, 1], color='orange', marker='o')
        plt.scatter(points[:10, 0], points[:10, 1], color='red', marker='o')
    else:
        class0_id_idx = (id_labels == 0).nonzero(as_tuple=True)[0].cpu().detach().numpy()
        features0 = smaller_features[class0_id_idx, :]
        class1_id_idx = (id_labels == 1).nonzero(as_tuple=True)[0].cpu().detach().numpy()
        features1 = smaller_features[class1_id_idx, :]
        centers = torch.transpose(net.m1/net.N1.unsqueeze(0), 0, 1).cpu().detach().numpy()
        svhn_smaller_features = svhn_smaller_features.cpu().detach().numpy()
        points = np.concatenate(
            (centers, features0.cpu().detach().numpy(), features1.cpu().detach().numpy(), svhn_smaller_features), axis=0)
        points = tsne.fit_transform(points)
        plt.scatter(points[2:features0.size(0) + 2, 0], points[2:features0.size(0) + 2, 1], color='blue',
                    marker='o')
        plt.scatter(points[features0.size(0) + 2:130, 0], points[features0.size(0) + 2:130, 1], color='blue',
                    marker='^')
        plt.scatter(points[130:, 0], points[130:, 1], color='orange', marker='o')
        plt.scatter(points[:2, 0], points[:2, 1], color='red', marker='o')


    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    #plt.show()
    plt.savefig("proposed_plot.pdf", format="pdf")

    # tsne = TSNE(n_components=2, random_state=42)
    # centers = np.array(get_centers(features, id_sub_labels))
    # # print(centers.shape)
    # points = np.concatenate(
    #     (centers, features0.cpu().detach().numpy(), features1.cpu().detach().numpy()), axis=0)
    # points = tsne.fit_transform(points)
    #
    # plt.scatter(points[10:features0.size(0) + 10, 0], points[10:features0.size(0) + 10, 1], color='blue', marker='o')
    # plt.scatter(points[features0.size(0) + 10:138, 0], points[features0.size(0) + 10:138, 1], color='blue', marker='^')
    # #plt.scatter(points[138:, 0], points[138:, 1], color='orange', marker='o')
    # plt.scatter(points[:10, 0], points[:10, 1], color='red', marker='o')
    # plt.title("t-SNE Visualization")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.show()
    #
    # tsne = TSNE(n_components=2, random_state=42)
    # points = np.concatenate((pred_probs.cpu().detach().numpy(), svhn_pred_probs.cpu().detach().numpy()), axis=0)
    # #points = tsne.fit_transform(points)
    # # centers = (centers - np.min(centers))/(np.max(centers) - np.min(centers))*2-1
    # # features = tsne.fit_transform(features)
    # # features = (features - np.min(features)) / (np.max(features) - np.min(features)) * 2 - 1
    #
    # plt.scatter(points[:128, 0], points[:128, 1])
    # plt.scatter(points[128:, 0], points[128:, 1])
    # plt.title("Output Visualization")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.show()