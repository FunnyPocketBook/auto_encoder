import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda') # Selects GPU
torch.set_num_threads(1) # Per default all threads

# DL01 ab 69, DL02 42-83, 93-152
# https://ml-cheatsheet.readthedocs.io/en/latest/index.html
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
# https://ml-cheatsheet.readthedocs.io/en/latest/layers.html
# https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html
# https://ruder.io/optimizing-gradient-descent/index.html
# extend network such that on 10 images the output is correct. Then to 100
# at the beginning, only do up to 1000 bc if that's good, 50k are also going to be good
# if the network is grey or black and stuck, as the god
# data loading and preprocessing
# tbd :)
# we want a numpy matrix with shape (?, 32, 32, 3)

# Read data
def deserialize(path):
    with open(path, 'rb') as file:
        return pickle.load(file, encoding='bytes')

# Reshape the matrix from a (10000, 3072) matrix to a (10000, 3, 32, 32) matrix
def reshape(matrix):
    matrix = matrix.reshape((10000, 3, 1024))
    matrix = matrix.reshape((10000, 3, 32, 32))
    return np.moveaxis(matrix, 1, -1) # Move channel dimension to the last dimension

def preprocess():
    data = reshape(deserialize("cifar-10-batches-py/data_batch_1")[b'data']) # Load data of batch 1
    for i in range(2,5): # Add data of batches 2-5 into data
        batch = reshape(deserialize("cifar-10-batches-py/data_batch_" + str(i))[b'data'])
        # data is in format (10000, 3072)
        data = np.append(data, batch, axis=0)
    
    return np.float32(data / 255)


# pytorch model must always define all layers in constructor
class Model(nn.Module):
    def __init__(self, width, height, channels):
        super(Model, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels

        # encode stage
        # input channels, output channels, kernel (2,2), maybe stride (2,2)
        self.convolutional_0 = nn.Conv2d(3, 5, 8, 2)
        self.linear_0 = nn.Linear(45, 10)

        # decode stage
        self.linear_1 = nn.Linear(10, 36)
        self.deconvolutional_0 = nn.ConvTranspose2d(4, 6, 4, 2)
        self.deconvolutional_1 = nn.ConvTranspose2d(6, 3, 11, 3)

    def encoder(self, x):
        # Change channels from 3 to 5 and apply kernel with stride
        x = F.leaky_relu(self.convolutional_0(x)) # Shape: [1, 3, 32, 32]
        # Reduce size of matrix by getting the maximum value of every 4x4 matrix
        x = F.max_pool2d(x, 4) # Shape: [1, 5, 13, 13]
        # Flatten the 4d matrix into 1d
        x = torch.flatten(x) # Shape: [1, 5, 3, 3]
        # Reduce vector to 10 dimensions
        x = F.leaky_relu(self.linear_0(x)) # Shape: [45]
        # Shape: [10]
        return x
    
    # last linear torch.sigmoid instead of leaky_relu
    def decoder(self, x):
        x = F.leaky_relu(self.linear_1(x)) # Shape: [10]
        x = torch.reshape(x, (1, 4, 3, 3)) # Shape: [36]
        x = F.leaky_relu(self.deconvolutional_0(x)) # Shape: [1, 6, 8, 8]
        x = F.leaky_relu(self.deconvolutional_1(x)) # Shape: [1, 3, 32, 32]
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Shows input and output image
def show_image(original, trained):
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(original)
    axes[1].imshow(trained)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

# Actually training the data
def training(model, training_data):
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Function for getting the mean square error loss
    criterion = nn.MSELoss()

    iterations = 0
    running_loss = 0
    epoch = 0
    max_epoch = 200

    while (epoch < max_epoch):
        for idx, elem in enumerate(training_data):
            iterations += 1
            # Create tensor from matrix
            tensor = torch.from_numpy(elem).to(device)
            # has shape (32, 32, 3)
            # we want (1, 3, 32, 32)
            # Change dimension positions
            target = tensor.permute(2, 0, 1).unsqueeze(0)
            # Reset optimizer for each data
            optimizer.zero_grad()
            output = model(target)
            loss = criterion(output, target)
            # Track loss
            running_loss += loss.item()
            print("loss: {0: .4f}, epoch: {1}, iterations: {2}"
                .format(running_loss / iterations, epoch, iterations))
            # Backtracking
            loss.backward()
            optimizer.step()
        epoch += 1
    
    # Reconstructing original image for showing it
    original = training_data[0]
    model.eval()
    reconstructed = torch.tensor(elem, requires_grad=True).to(device)
    reconstructed = tensor.permute(2, 0, 1).unsqueeze(0)
    reconstructed = model(reconstructed)
    reconstructed = np.moveaxis(reconstructed.detach().cpu().numpy()[0], 0, 2)
    show_image(original, reconstructed)


a = preprocess()
auto_encoder = Model(32, 32, 3)
auto_encoder.cuda(device)
training(auto_encoder, a[0:1])