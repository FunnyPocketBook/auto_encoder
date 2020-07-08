from scipy.ndimage import gaussian_filter
import autoencoder
import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt

def apply_gaussian_blur(images, sigma):
    result = []
    for idx, img in enumerate(images):
        result.append(gaussian_filter(img, sigma=sigma))
        print(f"\rApplied filter to {idx+1:>{len(str(len(images)))}}/{len(images)} images.", end="")
    print("")
    return result

def apply_vert_wave(images, shift):
    result = []
    for idx, img in enumerate(images):
        rows, cols, channels = img.shape
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(shift * np.sin(9.0 * np.pi * i / 180))
                offset_y = 0
                if j+offset_x < rows:
                    img_output[i,j] = img[i, (j+offset_x)%cols]
                else:
                    img_output[i,j] = 0
        result.append(img_output)
        print(f"\rApplied distortion to {idx+1:>{len(str(len(images)))}}/{len(images)} images.", end="")
    print("")
    return result

def load_distorted_imgs(images, type, path, shift):
    if type == "gaussian":
        function = apply_gaussian_blur
    elif type == "vert_wave":
        function = apply_vert_wave
    else:
        raise ValueError("Only \"gaussian\" or \"vert_wave\" allowed.")
    directory = "stability/dist_imgs"
    if not os.path.exists(directory):
            os.makedirs(directory)
    try:
        distorted_imgs_file = open(path, "rb")
        print("Loading distorted images.")
        distorted_imgs = pickle.load(distorted_imgs_file)
        distorted_imgs_file.close()
        print("Finished loading distorted images.")
    except FileNotFoundError:
        print("Applying filter to all images.")
        distorted_imgs = function(images, shift)
        print("Finished applying filter to all images.")
        distorted_imgs_file = open(path, "wb")
        pickle.dump(distorted_imgs, distorted_imgs_file)
        distorted_imgs_file.close()
    return distorted_imgs

def show_imgs(orig_img, filter_img, name):
    fig, axes = plt.subplots(2, len(filter_img))
    for i in range(0, len(filter_img)):
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        axes[0, i].imshow(orig_img[i])
        axes[1, i].imshow(filter_img[i])
    plt.tight_layout(w_pad=1, h_pad=10)
    fig.set_size_inches(20, 5)
    directory = "stability"
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(f"{directory}/{name}.png")
    plt.clf()
    plt.close()

ae = autoencoder.Model(32, 32, 3)
ae.cuda(autoencoder.device)
data = autoencoder.preprocess()
test = autoencoder.load_test_batch()
gaussian_training_img = load_distorted_imgs(data, "gaussian", "stability/dist_imgs/gaussian_training_img.bin", 1)
gaussian_test_img = load_distorted_imgs(test, "gaussian", "stability/dist_imgs/gaussian_test_img.bin", 1)
vert_wave_training_img = load_distorted_imgs(data, "vert_wave", "stability/dist_imgs/vert_wave_training_img.bin", 5)
#vert_wave_test_img = load_distorted_imgs(test, "vert_wave", "stability/dist_imgs/vert_wave_test_img.bin", 15)

#show_imgs(data[:10], gaussian_training_img[:10], "gaussian")
#show_imgs(data[:10], vert_wave_training_img[:10], "vert_wave")
autoencoder.training(ae, gaussian_training_img[:1000], test, data)