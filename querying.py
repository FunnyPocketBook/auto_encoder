import torch
import autoencoder
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def minkowski_like_dist(fstp, sndp, p):
    ary = sum(abs(fstp[i] - sndp[i])**p for i in range(0, len(fstp)))
    result = ary**(1/p)
    return result

def cosine_dist(fstp, sndp):
    numerator = sum(fstp[i] * sndp[i] for i in range(0, len(fstp)))
    denom_fst = sum(fstp[i] ** 2 for i in range(0, len(fstp)))
    denom_snd = sum(sndp[i] ** 2 for i in range(0, len(sndp))) 
    return 1 - (numerator / (math.sqrt(denom_fst) * math.sqrt(denom_snd)))

def encode(auto_encoder, data):
    imgs = []   
    encoder = auto_encoder.encoder
    for i, data_img in enumerate(data):
        tensor = torch.from_numpy(data_img).to(device)
        target = tensor.permute(2, 0, 1).unsqueeze(0)
        img = encoder(target).detach().cpu().numpy()
        imgs.append(img)
        if i+1 % 10 == 0:
            print(f"\rEncoded {i}/{len(data)}", end="")
    print("")
    return imgs

def distance(p, query_img_enc, org_imgs_enc, auto_encoder):
    dist = []
    for i, org_img in enumerate(org_imgs_enc):
        dist.append((minkowski_like_dist(query_img_enc, org_img, p), i))
    similar_imgs = sorted(dist, key=lambda x: x[0])
    return similar_imgs[1:11]

def show_similar_imgs(query_img, similar_imgs, auto_encoder, p, idx):
    auto_encoder.eval()
    fig, axes = plt.subplots(2, len(similar_imgs))
    for i in range(0, len(similar_imgs)):
        original = query_img
        index = similar_imgs[i][1]
        elem = data[index]
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        axes[0, i].imshow(original)
        axes[1, i].imshow(elem)
    plt.tight_layout(w_pad=1, h_pad=10)
    fig.set_size_inches(20, 5)
    directory = f"similarity/{p}"
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(f"{directory}/similarity_{p}_{idx}.png")
    plt.clf()
    plt.close()

ae = autoencoder.Model(32, 32, 3)
ae.cuda(autoencoder.device)
device = torch.device('cuda')
data = autoencoder.preprocess()
test = autoencoder.load_test_batch()
model_name = "model_mse_lr0.0006_n50000.pt"
checkpoint = torch.load("model/"+model_name, map_location="cpu")
ae.load_state_dict(checkpoint["model_state_dict"])
ae.eval()
query_img_idx = [1, 245, 7432, 234, 6783, 8992, 5102, 3, 4568, 1003, 5555, 543, 789, 3758, 2134, 6201, 9, 9123, 923, 376]
query_img = [test[x] for x in query_img_idx]
query_img_enc = encode(ae, query_img)

try:
    enc_imgs = open('enc_imgs.bin', 'rb')
    print("loading encoded images")
    encoded_images = pickle.load(enc_imgs)
    enc_imgs.close()
except FileNotFoundError:
    print("encoding all images")
    encoded_images = encode(ae, data)
    print("finished encoding all images")
    enc_imgs = open('enc_imgs.bin', 'wb')
    pickle.dump(encoded_images, enc_imgs)
    enc_imgs.close()

try:
    enc_imgs = open('enc_test.bin', 'rb')
    print("loading encoded test images")
    query_imgs_enc = pickle.load(enc_imgs)
    enc_imgs.close()
except FileNotFoundError:
    print("encoding all test images")
    query_imgs_enc = encode(ae, test)
    print("finished encoding all test images")
    enc_imgs = open('enc_test.bin', 'wb')
    pickle.dump(query_imgs_enc, enc_imgs)
    enc_imgs.close()

query_img_assignment = data[0]
query_img_assignment_enc = encode(ae, [query_img_assignment])

for p in [1,2,100]:
    d = distance(p, query_img_assignment_enc[0], encoded_images, ae)
    show_similar_imgs(query_img_assignment, d, ae, p, 0)

def cosine_distance(query_img_enc, org_imgs_enc, auto_encoder):
    dist = []
    for i, org_img in enumerate(org_imgs_enc):
        dist.append((cosine_dist(query_img_enc, org_img), i))
    similar_imgs = sorted(dist, key=lambda x: x[0])
    return similar_imgs[1:11]
for i, index in enumerate(query_img_idx):
    d = cosine_distance(query_img_enc[i], encoded_images, ae)
    show_similar_imgs(query_img[i], d, ae, "cosine", index)