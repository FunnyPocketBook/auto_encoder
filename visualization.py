import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random as rnd
import seaborn as sns
import autoencoder
import umap

def gen_random_numbers(amt, max):
    count = 0
    while True:
        count += 1
        if (count > amt): return
        yield rnd.randint(0, max)

def generate_pairplot(auto_encoder, data):
    imgs = []
    encoder = auto_encoder.encoder
    random_numbers = gen_random_numbers(50, len(data))
    for idx, i in enumerate(random_numbers):
        tensor = torch.from_numpy(data[i]).to(torch.device('cuda'))
        target = tensor.permute(2, 0, 1).unsqueeze(0)
        img = encoder(target).detach().cpu().numpy()
        imgs.append(img)
        print(f"\rProcessing {idx+1:>2}/50 images for pairplot.", end="")
    print("\nGenerating pairplot...")
    dataframe = pd.DataFrame(data=np.array(imgs), columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"])
    g = sns.pairplot(dataframe)
    plt.tight_layout()
    plt.savefig("splom_{0}.png".format(model_name))
    print("Generated pairplot.")
    plt.clf()

def generate_scatterplot(auto_encoder, data, labels):
    imgs = []
    encoder = auto_encoder.encoder
    reducer = umap.UMAP()
    for idx, i in enumerate(data):
        tensor = torch.from_numpy(i).to(torch.device('cuda'))
        target = tensor.permute(2, 0, 1).unsqueeze(0)
        img = encoder(target).detach().cpu().numpy()
        imgs.append(img)
        print(f"\rProcessing {idx+1:>{len(str(len(data)))}}/{len(data)} images for scatterplot.", end="")
    print("\nGenerating scatterplot...")
    embedding = reducer.fit_transform(imgs)
    print(embedding.shape)
    fig = plt.figure()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in labels]
    )
    fig.set_size_inches(15, 15)
    plt.tight_layout()
    plt.savefig("umap_{0}.png".format(model_name))
    print("Generated scatterplot.")
    plt.clf()

ae = autoencoder.Model(32, 32, 3)
ae.cuda(autoencoder.device)
data = autoencoder.preprocess()
test = autoencoder.load_test_batch()
labels, test_labels = autoencoder.load_labels()
model_name = "model.hdf5"
checkpoint = torch.load("model/"+model_name, map_location="cpu")
ae.load_state_dict(checkpoint["model_state_dict"])
ae.eval()
generate_pairplot(ae, test)
generate_scatterplot(ae, test, test_labels)