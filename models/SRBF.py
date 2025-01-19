import random
import torch
import torch.nn as nn
from sklearn.cluster import KMeans#, AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_score
#from sklearn.mixture import GaussianMixture
from utils.my_utils import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

training_device = "cuda:0"
dataset="human"

import matplotlib.pyplot as plt

class SRBF(nn.Module):

    def __init__(
        self,
        feature_extractor,
        classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
        device,
        dataset_,
        binary_map
    ):
        super().__init__()
        global training_device
        global dataset
        dataset = dataset_
        self.binary_map = binary_map

        training_device = device
        self.mask = torch.tensor(binary_map).to(training_device)
        self.gamma = gamma
        self.classes = classes
        self.feature_extractor = feature_extractor
        self.subclass_found, self.clusters_found = [], []
        self.post_subclass = False

        # --- SRBF ---
        self.W1_part1 = nn.Parameter(torch.zeros(model_output_size, centroid_size))
        nn.init.kaiming_normal_(self.W1_part1, nonlinearity="relu")
        self.num_centroids = classes
        num_centroids = self.num_centroids
        self.W1_part2 = nn.Parameter(torch.zeros(centroid_size, num_centroids))
        nn.init.kaiming_normal_(self.W1_part2, nonlinearity="relu")
        self.register_buffer("N1", torch.zeros(num_centroids) + 13)
        self.register_buffer("m1", torch.normal(torch.zeros(centroid_size, num_centroids), 0.05))
        self.m1 = self.m1 * self.N1
        self.sigma1 = nn.Parameter(torch.full((num_centroids,), length_scale, device=training_device))


    def srbf(self, x):
        if not self.post_subclass:
            features = torch.einsum('ij,jk->ik', x, self.W1_part1)  # from (batch_size, model_output_size) to (batch_size,centroid_size)
            z = torch.einsum('ij,jk->ijk', features, self.W1_part2) # from (batch_size,centroid_size) to (batch_size,centroid_size, classes)
            embeddings = self.m1 / self.N1.unsqueeze(0)
            diff = z - embeddings.unsqueeze(0)
            diff = (diff ** 2).mean(1).div(2 * self.sigma1 ** 2).mul(-1).exp()
        elif self.post_subclass:
            z = torch.einsum("ij,mnj->imn", x, self.W_s)
            features = z
            embeddings = self.m1 / self.N1.unsqueeze(0)
            diff = z - embeddings.unsqueeze(0)
            diff = (diff ** 2).mean(1).div(2 * self.sigma1 ** 2).mul(-1).exp()

        return diff, features

    def forward(self, x):
        if not self.post_subclass:
            features = self.feature_extractor(x)
            y_pred, features1 = self.srbf(features)
            return y_pred, y_pred, features1, features
        elif self.post_subclass:

            features = self.feature_extractor(x)
            sub_y_pred, features1 = self.srbf(features)
            selected_values0 = torch.masked_select(sub_y_pred, self.mask == 0).view(-1,int(len(self.mask) - torch.sum(self.mask)))
            selected_values1 = torch.masked_select(sub_y_pred, self.mask == 1).view(-1, int(torch.sum(self.mask)))
            max_values0, _ = torch.max(selected_values0, dim=1)
            max_values1, _ = torch.max(selected_values1, dim=1)

            y_pred = torch.cat((max_values0.view(-1, 1), max_values1.view(-1, 1)), dim=1)
            return y_pred, sub_y_pred, features1, features



    def update_embeddings(self, x, y, freeze):

        if not self.post_subclass:
            # --- Update of SRBF ---
            self.N1 = self.gamma * self.N1 + (1 - self.gamma) * y.sum(0)
            features = self.feature_extractor(x)
            z = torch.einsum('ij,jk->ik', features, self.W1_part1)
            z1 = torch.einsum('ij,jk->ijk', z, self.W1_part2)
            embedding_sum1 = torch.einsum("ijk,ik->jk", z1, y)
            self.m1 = self.gamma * self.m1 + (1 - self.gamma) * embedding_sum1

        elif self.post_subclass:
            self.N1 = self.gamma * self.N1 + (1 - self.gamma) * y.sum(0)
            features = self.feature_extractor(x)
            z = torch.einsum("ij,mnj->imn", features, self.W_s)
            embedding_sum1 = torch.einsum("ijk,ik->jk", z, y)
            self.m1 = self.gamma * self.m1 + (1 - self.gamma) * embedding_sum1

    def criterion(self, x, y, s_y):

        global dataset
        global training_device
        new_sigma = []
        self.post_subclass = True
        num_clusters = 200 if dataset == "tiny_imagenet" else 10
        if dataset == "human":
            num_clusters = 7
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(x)
        labels = torch.tensor(kmeans.labels_)
        centers = kmeans.cluster_centers_
        total_label_count = y.bincount()
        # print(total_label_count)
        # exit()
        subclasses = []
        for cluster_id in range(num_clusters):
            curr_subclass_idx = (labels == cluster_id).nonzero(as_tuple=True)[0]
            subclass_items = torch.flatten(torch.index_select(x, 0, curr_subclass_idx), start_dim=1)
            distances = pairwise_distances(torch.tensor(centers[cluster_id]).double().unsqueeze(0),
                                           torch.tensor(subclass_items.numpy()).double())
            new_sigma.append(torch.mean(distances))
            cluster_labels = y[curr_subclass_idx]
            if dataset == "human":
                most_common_label = (cluster_labels.bincount() / total_label_count).argmax().item()
                print(f"cluster: {cluster_id} | most common label: {most_common_label}")

            if dataset == "human":
                subclasses.append(most_common_label)
        if dataset == "human":
            self.mask = torch.tensor(subclasses).to(training_device)
            print(self.mask)
        self.update_model(np.array(centers), 0)

        new_sigma = np.mean(np.array(new_sigma))

        if dataset == "mnist":
            self.sigma1 = nn.Parameter(torch.ones(10).to(device=training_device)*new_sigma*0.1)
        elif dataset == "FM":
            self.sigma1 = nn.Parameter(torch.ones(10).to(device=training_device) * new_sigma * 0.5)
        elif dataset == "cifar":
            self.sigma1 = nn.Parameter(torch.ones(10).to(device=training_device))
        elif dataset == "svhn":
            self.sigma1 = nn.Parameter(torch.ones(10).to(device=training_device)*new_sigma*0.1)
        elif dataset == "tiny_imagenet":
            self.sigma1 = nn.Parameter(torch.ones(200).to(device=training_device)*new_sigma*0.5)
        elif dataset == "human":
            self.sigma1 = nn.Parameter(torch.ones(len(subclasses)).to(device=training_device)*0.2)
        return

    def update_model(self, cluster_centers, class_id):
        self.update_rbf_layers(class_id, cluster_centers)

    def update_rbf_layers(self, class_id, cluster_centers):
        # convert centers of size e.g. (512,10) to (512,11) by duplicating "class id"-th center and
        # moving the centers towards the cluster centers of the features
        global training_device
        new_centers = np.zeros((cluster_centers.shape[1], cluster_centers.shape[0]))
        for i, cluster_center in enumerate(cluster_centers):
            new_centers[:, i] = cluster_center

        self.m1 = torch.tensor(new_centers).to(training_device)

        # W expanded from (model_output_size, centroid_size) to (centroid_size, num_classes, model_output_size)
        self.W_s = nn.Parameter(self.W1_part1.transpose(0,1).unsqueeze(1).repeat(1,self.m1.size(1),1).to(training_device))

        N1 = self.N1
        for _ in range(cluster_centers.shape[0] - 2):
            N1 = torch.cat((N1[:class_id], N1[class_id].unsqueeze(0), N1[class_id:]))
        self.register_buffer("N1", N1)
