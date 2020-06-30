import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda') # Selects GPU
torch.set_num_threads(1) # Per default all threads
model_path = "model/model1.pt"
learning_rate = 0.0005
loss_function = "mse"

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
        self.convolutional_0 = nn.Conv2d(3, 13, 1, 1)
        self.convolutional_1 = nn.Conv2d(13, 26, 3, 2)
        self.convolutional_2 = nn.Conv2d(26, 38, 2, 2)
        self.convolutional_3 = nn.Conv2d(38, 44, 2, 2)
        self.linear_0 = nn.Linear(396, 10)

        # decode stage
        self.linear_1 = nn.Linear(10, 16)
        self.deconvolutional_0 = nn.ConvTranspose2d(1, 3, 2)
        self.deconvolutional_1 = nn.ConvTranspose2d(3, 6, 2, 2)
        self.deconvolutional_2 = nn.ConvTranspose2d(6, 9, 3, 2)
        self.deconvolutional_3 = nn.ConvTranspose2d(9, 10, 2)
        self.linear_2 = nn.Linear(self.get_decode_lin_shape(), 3072)

    def get_decode_lin_shape(self):
        x = torch.randn([1, 1, 4, 4])
        x = self.decode_conv(x)
        return x.size()[0]

    def encoder(self, x):
        # Change channels from 3 to 5 and apply kernel with stride
        x = F.leaky_relu(self.convolutional_0(x)) # Shape: [1, 13, 32, 32]
        x = F.leaky_relu(self.convolutional_1(x)) # Shape: [1, 26, 16, 16]
        x = F.leaky_relu(self.convolutional_2(x)) # Shape: [1, 38, 8, 8]
        x = F.leaky_relu(self.convolutional_3(x)) # Shape: [1, 44, 4, 4]
        # Reduce size of matrix by getting the maximum value of every 4x4 matrix
        # x = F.max_pool2d(x, 4) # Shape: [1, 10, 7, 7]
        # Flatten the 4d matrix into 1d
        x = torch.flatten(x) # Shape: [48]
        # Reduce vector to 10 dimensions
        x = F.leaky_relu(self.linear_0(x)) # Shape: [10]
        return x
    
    # last linear torch.sigmoid instead of leaky_relu
    def decoder(self, x):
        #x = torch.reshape(x, (1, 1, 1, 10))
        x = F.leaky_relu(self.linear_1(x)) # Shape: [16]
        x = torch.reshape(x, (1, 1, 4, 4))
        x = self.decode_conv(x)
        x = torch.sigmoid(self.linear_2(x))
        x = torch.reshape(x, (1, 3, 32, 32))
        return x

    def decode_conv(self, x):
        x = F.leaky_relu(self.deconvolutional_0(x)) # Shape: [1, 3, 5, 5]
        x = F.leaky_relu(self.deconvolutional_1(x)) # Shape: [1, 5, 10, 10]
        x = F.leaky_relu(self.deconvolutional_2(x)) # Shape: [1, 7, 21, 21]
        x = F.leaky_relu(self.deconvolutional_3(x)) # Shape: [1, 9, 22, 22]
        x = torch.flatten(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Shows input and output image
def show_image(model, training_data, epoch, loss, n, start_time, model_name, save=False):
    model.eval()
    fig, axes = plt.subplots(2, len(training_data))
    for i in range(0, len(training_data)):
        original = training_data[i]
        elem = training_data[i]
        fig.suptitle("Comparison, loss: {0:.6f}".format(loss))
        tensor = torch.from_numpy(elem).to(device)
        trained = torch.tensor(elem, requires_grad=True).to(device)
        trained = tensor.permute(2, 0, 1).unsqueeze(0)
        trained = model(trained)
        trained = np.moveaxis(trained.detach().cpu().numpy()[0], 0, 2)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        axes[0, i].imshow(original)
        axes[1, i].imshow(trained)
    plt.tight_layout(w_pad=1, h_pad=10)
    fig.set_size_inches(20, 5)
    if save:
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = "result/{0}".format(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig("{4}/{0}_result_ep{1}_l{2:.6f}_n{3}.png".format(time, epoch, loss, n, directory))
    else:
        plt.show()
    plt.clf()
    plt.close()


def save_model(path, epoch, iterations, model, optimizer, loss, history_loss):
    print("\nSaving model...")
    torch.save({
        "epoch": epoch,
        "iterations": iterations,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history_loss": history_loss
    }, path+".pt")
    print("Model saved")
    if epoch % 50 == 0:
        print("Creating backup...")
        torch.save({
            "epoch": epoch,
            "iterations": iterations,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "history_loss": history_loss
        }, path+"_e{0}.pt".format(epoch))
        print("Backup created")


# Actually training the data
def training(model, training_data):
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Function for getting the mean square error loss
    # BCELoss 0.005 too high, loss is increasing again.
    # BCELoss 0.002 fine for 10 images
    # MSELoss 0.001 fine for 10 images, but grey with 100, even if the dense layer has been reduced
    # MSELoss 0.003 kinda fine for 1000 images but there are still some artifacts. Not sure why, maybe too many in_channels in encoding stage
    if loss_function == "mse":
        criterion = nn.MSELoss()
    elif loss_function == "bce":
        criterion = nn.BCELoss()

    iterations = 0
    running_loss = 0
    history_loss = []
    epoch = 0
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_note = ""
    model_name = "model_{0}_lr{1}_n{2}{3}".format(loss_function, learning_rate, len(training_data), model_note)
    model_path = "model/{0}".format(model_name)

    try:
        checkpoint = torch.load(model_path+".pt")
        epoch = checkpoint["epoch"]
        iterations = checkpoint["iterations"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        running_loss = checkpoint["loss"]
        history_loss = checkpoint["history_loss"]
        print("Using previous model")
    except FileNotFoundError:
        print("no model found")

    try:
        while True:
            for idx, elem in enumerate(training_data):
                iterations += 1
                # Create tensor from matrix
                tensor = torch.from_numpy(elem).to(device)
                # Change dimension positions
                target = tensor.permute(2, 0, 1).unsqueeze(0)
                # Reset optimizer for each data
                optimizer.zero_grad()
                output = model(target)
                loss = criterion(output, target)
                history_loss.append(loss)
                # Track loss
                running_loss += loss.item()
                print("\rloss: {0: .6f}, epoch: {1}, iterations: {2}"
                    .format(running_loss / iterations, epoch, iterations), end='')
                if iterations % 1000 == 0:
                    show_image(model, training_data[20:30], epoch, running_loss/iterations, len(training_data), start_time, model_name, save=True)
                # Backtracking
                loss.backward()
                optimizer.step()
            epoch += 1
            save_model(model_path, epoch, iterations, model, optimizer, running_loss, history_loss)
        show_image(model, training_data[20:30], epoch, running_loss/iterations, len(training_data), start_time, model_name, save=True)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        show_image(model, training_data[20:30], epoch, running_loss/iterations, len(training_data), start_time, model_name, save=True)
        save_model(model_path, epoch, iterations, model, optimizer, running_loss, history_loss)

a = preprocess()
auto_encoder = Model(32, 32, 3)
auto_encoder.cuda(device)
training(auto_encoder, a[0:1000])