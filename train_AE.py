import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils.progress import progress_bar
from datasets_config import cifar_trainset, cifar_train_loader, cifar_test_loader, cifar_test_set, cifar_val_loader, cifar_valset
from datasets_config import train_svhn,svhn_testset, svhn_train_loader, svhn_val_loader, svhn_test_loader, svhn_valset, svhn_trainset
from datasets_config import fashionmnist, FM_train_loader, FM_val_loader, FM_test_loader, FM_trainset, FM_valset
from datasets_config import MNIST_trainloader, MNIST_validationloader, MNIST_testloader, testset, valset
from datasets_config import get_tiny_image_net, get_human
from models.autoencoders import CAE, CAE2, CAE3, CAE5
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', choices=["mnist", "cifar", "tiny_imagenet", "svhn", "FM", "human"], required=True,
        help="Which dataset to use for training")
args = parser.parse_args()
dataset = args.dataset

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 30
# Define Autoencoder model
if dataset == "mnist":
    trainloader = MNIST_trainloader
    latent_dim = 64
    autoencoder = CAE(latent_dim).to(device)
elif dataset == "cifar":
    trainloader = cifar_train_loader
    latent_dim = 512
    autoencoder = CAE2(latent_dim).to(device)
elif dataset == "svhn":
    trainloader = svhn_train_loader
    latent_dim = 512
    autoencoder = CAE2(latent_dim).to(device)
elif dataset == "FM":
    trainloader = FM_train_loader
    latent_dim = 256
    autoencoder = CAE(latent_dim).to(device)
elif dataset == "tiny_imagenet":
    _, _, tiny_train_loader, _ = get_tiny_image_net()
    trainloader = tiny_train_loader
    latent_dim = 512
    autoencoder = CAE3(latent_dim).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop

losses = []


def plot_images(original, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for i in range(5):
        axes[0, i].imshow(original[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        axes[0, i].set_title('Ground Truth')

        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    plt.tight_layout()
    plt.show()


for epoch in range(num_epochs):
    running_loss = 0.0
    training_batches = 0
    for i, data in enumerate(trainloader, 0):
        # Get inputs
        inputs, _ = data
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(inputs)

        # Compute loss
        loss = criterion(outputs, inputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.clone().cpu().item()
        training_batches += 1
        progress_bar(i, len(trainloader),
                 f'Epoch: {epoch} | Class Loss: {running_loss / training_batches:.4f} |')

    if epoch == num_epochs-1:
        plot_images(inputs[:5].cpu().detach(), outputs[:5].cpu().detach())
    losses.append(running_loss / training_batches)

print("Training finished.")


# Save the trained model

if dataset == "mnist":
    torch.save(autoencoder.state_dict(), "autoencoder_mnist.pth")
elif dataset == "cifar":
    torch.save(autoencoder.state_dict(), "autoencoder_cifar.pth")
elif dataset == "svhn":
    torch.save(autoencoder.state_dict(), "autoencoder_svhn.pth")
elif dataset == "FM":
    torch.save(autoencoder.state_dict(), "autoencoder_fm.pth")
elif dataset == "tiny_imagenet":
    torch.save(autoencoder.state_dict(), "autoencoder_tiny_imagenet.pth")
elif dataset == "human":
    torch.save(autoencoder.state_dict(), "autoencoder_human18.pth")


def plot_learning_curve(losses):
    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


plot_learning_curve(losses)