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
learning_rate = 0.0006
loss_function = "mse"

# DL01 ab 69, DL02 42-83, 93-152
# https://ml-cheatsheet.readthedocs.io/en/latest/index.html
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
# https://ml-cheatsheet.readthedocs.io/en/latest/layers.html
# https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html
# https://ruder.io/optimizing-gradient-descent/index.html

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
    for i in range(2,6): # Add data of batches 2-5 into data
        batch = reshape(deserialize("cifar-10-batches-py/data_batch_" + str(i))[b'data'])
        # data is in format (10000, 3072)
        data = np.append(data, batch, axis=0)
    return np.float32(data / 255)

def load_labels():
    labels = deserialize("cifar-10-batches-py/data_batch_1")[b'labels'] # Load data of batch 1
    for i in range(2,6): # Add data of batches 2-5 into data
        batch = deserialize("cifar-10-batches-py/data_batch_" + str(i))[b'labels']
        # data is in format (10000, 3072)
        labels = np.append(labels, batch, axis=0)
    test_labels = deserialize("cifar-10-batches-py/test_batch")[b'labels']
    return labels, test_labels

def load_test_batch():
    return np.float32(reshape(deserialize("cifar-10-batches-py/test_batch")[b'data']) / 255)

class RingBuffer():
    """
   I stole this from Samuel Schiegg (with permission)
   """
    def __init__(self, max_length, reactivity_factor=10):
        self.max_length = max_length
        self.index = 0
        self.rb = []
        self.intermed = []
        self.reactivity_factor = reactivity_factor
 
    def add(self, x):
        if len(self.rb) < self.max_length:
            self.rb.append(x)
            self.index += 1
        else:
            if len(self.intermed) < self.__stability():
                self.intermed.append(x)
            else:
                x = sum(self.intermed) / len(self.intermed)
                if self.index >= self.max_length:
                    self.index = 0
                self.rb[self.index] = x
                self.index += 1        
                self.intermed.clear()
 
    def __stability(self):
        if self.reactivity_factor > self.max_length:
            self.reactivity_factor = self.max_length
        return (int)(self.max_length / self.reactivity_factor)
 
    def clear(self):
        self.rb = []
        self.index = 0
 
    def avg(self):
        return sum(self.rb) / len(self.rb)


# pytorch model must always define all layers in constructor
class Model(nn.Module):
    def __init__(self, width, height, channels):
        super(Model, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels

        # encode stage
        # input channels, output channels, kernel (2,2), maybe stride (2,2)
        self.convolutional_0 = nn.Conv2d(3, 13, 1)
        self.convolutional_1 = nn.Conv2d(13, 26, 2, 2)
        self.convolutional_2 = nn.Conv2d(26, 38, 2, 2)
        self.convolutional_3 = nn.Conv2d(38, 45, 2, 3)
        self.linear_0 = nn.Linear(405, 10)

        # decode stage
        self.linear_1 = nn.Linear(10, 16)
        self.deconvolutional_0 = nn.ConvTranspose2d(1, 5, 2, 1)
        self.deconvolutional_1 = nn.ConvTranspose2d(5, 9, 2, 2)
        self.deconvolutional_2 = nn.ConvTranspose2d(9, 14, 3, 2)
        self.deconvolutional_3 = nn.ConvTranspose2d(14, 19, 2, 1)
        self.linear_2 = nn.Linear(self.get_decode_lin_shape(), 3072)

    def get_decode_lin_shape(self):
        x = torch.randn([1, 1, 4, 4])
        x = self.decode_conv(x)
        return x.size()[0]

    def encoder(self, x):
        # Change channels from 3 to 5 and apply kernel with stride
        x = F.leaky_relu(self.convolutional_0(x)) # Shape: [1, 13, 32, 32]
        x = F.leaky_relu(self.convolutional_1(x)) # Shape: [1, 26, 26, 26]
        x = F.leaky_relu(self.convolutional_2(x)) # Shape: [1, 38, 8, 8]
        x = F.leaky_relu(self.convolutional_3(x)) # Shape: [1, 45, 3, 3]
        x = torch.flatten(x) # Shape: [48]
        x = F.leaky_relu(self.linear_0(x)) # Shape: [10]
        return x
    
    def decoder(self, x):
        x = F.leaky_relu(self.linear_1(x)) # Shape: [16]
        x = torch.reshape(x, (1, 1, 4, 4))
        x = self.decode_conv(x)
        x = torch.sigmoid(self.linear_2(x))
        x = torch.reshape(x, (1, 3, 32, 32))
        return x

    def decode_conv(self, x):
        x = F.leaky_relu(self.deconvolutional_0(x)) # Shape: [1, 5, 5, 5]
        x = F.leaky_relu(self.deconvolutional_1(x)) # Shape: [1, 9, 10, 10]
        x = F.leaky_relu(self.deconvolutional_2(x)) # Shape: [1, 14, 21, 21]
        x = F.leaky_relu(self.deconvolutional_3(x)) # Shape: [1, 19, 22, 22]
        x = torch.flatten(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Shows input and output image
def show_image(model, training_data, epoch, loss, n, model_name, save=False, test=False):
    model.eval()
    fig, axes = plt.subplots(2, len(training_data))
    for i in range(0, len(training_data)):
        original = training_data[i]
        elem = training_data[i]
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
        if test:
            plt.savefig("{0}/result_test_ep{1}_n{2}.png".format(directory, epoch, n))
        else:
            plt.savefig("{4}/{0}_result_ep{1}_l{2:.6f}_n{3}.png".format(time, epoch, loss, n, directory))
    else:
        plt.show()
    plt.clf()
    plt.close()

def show_loss(history_loss, n, epoch, save=False):
    plt.plot(range(0,len(history_loss)), history_loss)
    if save:
        plt.savefig("loss/loss_ep{0}_n{1}_lr{2}.png".format(epoch, n, learning_rate))
    else:
        plt.show()
    plt.clf()
    plt.close()


def save_model(path, epoch, iterations, model, optimizer, loss, history_loss, current_index):
    torch.save({
        "epoch": epoch,
        "iterations": iterations,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history_loss": history_loss,
        "current_index": current_index
    }, path+".pt")


# Actually training the data
def training(model, training_data, test_data):
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

    ring_buffer = RingBuffer(1000)
    iterations = 0
    running_loss = 0
    history_loss = []
    epoch = 0
    current_index = 0
    model_note = ""
    model_name = "model_{0}_lr{1}_n{2}{3}".format(loss_function, learning_rate, len(training_data), model_note)
    #model_name = "model_mse_lr0.0005_n50000"
    model_path = "model/{0}".format(model_name)

    try:
        checkpoint = torch.load(model_path+".pt")
        epoch = checkpoint["epoch"]
        current_index = checkpoint["current_index"]
        iterations = checkpoint["iterations"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        running_loss = checkpoint["loss"]
        history_loss = checkpoint["history_loss"]
        print("Using previous model")
    except FileNotFoundError:
        print("no model found")

    start_time = datetime.datetime.now()
    progress = ""
    try:
        while True:
            for idx, elem in enumerate(training_data, start=current_index):
                iterations += 1
                current_index = idx+1
                time = datetime.datetime.now()
                # Create tensor from matrix
                tensor = torch.from_numpy(elem).to(device)
                # Change dimension positions
                target = tensor.permute(2, 0, 1).unsqueeze(0)
                # Reset optimizer for each data
                optimizer.zero_grad()
                output = model(target)
                loss = criterion(output, target)
                # Track loss
                running_loss += loss.item()
                loss_current = running_loss / iterations
                history_loss.append(loss_current)
                if iterations % 10000 == 0:
                    show_image(model, training_data[0:10], epoch, running_loss/iterations, len(training_data), model_name, save=True, test=False)
                # Backtracking
                loss.backward()
                optimizer.step()
                time = datetime.datetime.now() - time
                ring_buffer.add(time.microseconds)
                elapsed = datetime.datetime.now() - start_time
                elapsed = elapsed - datetime.timedelta(microseconds=elapsed.microseconds)
                remaining = datetime.timedelta(microseconds=ring_buffer.avg() * (len(training_data) - iterations%len(training_data)))
                remaining = remaining - datetime.timedelta(microseconds=remaining.microseconds)
                if iterations % 50 == 0:
                    progress = "\rTraining on n={0}: loss: {6:.6f} | iterations: {7: >8} | epoch: {5: >3} | {1: >6}/{0} | {2: >4}ms/iteration | elapsed: {3:>8} | remaining: {4:>8}" \
                        .format(len(training_data),
                            current_index,
                            int(ring_buffer.avg()/1000),
                            str(elapsed),
                            str(remaining), 
                            epoch, 
                            running_loss/iterations,
                            iterations)
                    print(progress, end="")
            epoch += 1
            save_model(model_path, epoch, iterations, model, optimizer, running_loss, history_loss, current_index)
            show_loss(history_loss, len(training_data), epoch, save=True)
            show_image(model, test_data[0:10], epoch, running_loss/iterations, len(training_data), model_name, save=True, test=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt")
        show_image(model, test_data[0:10], epoch, running_loss/iterations, len(training_data), model_name, save=True, test=True)
        show_image(model, training_data[0:10], epoch, running_loss/iterations, len(training_data), model_name, save=True, test=False)
        save_model(model_path, epoch, iterations, model, optimizer, running_loss, history_loss, current_index)
        show_loss(history_loss, len(training_data), epoch, save=True)