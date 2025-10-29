import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from model import autoencoderMLP4Layer

def get_two_ints():
    while True:
        try:
            user_input = int(input("Enter first int 0-59999\n"))
            if not (0 <= user_input <= 59999):
                print("Number not in range")
                continue  # go back to start of loop

            user_input2 = int(input("Enter second int 0-59999\n"))
            if not (0 <= user_input2 <= 59999):
                print("Number not in range")
                continue  # go back to start of loop

            print("Both indices are correct")
            return user_input, user_input2

        except ValueError:
            print("Invalid, enter a number")


def interpolate_between_imgs(model, img1, img2):
    n_steps = 8
    z1 = model.encode(img1)
    z2 = model.encode(img2)

    alphas = torch.linspace(0, 1, n_steps)

    interpolations = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        recon = model.decode(z_interp)
        interpolations.append(recon.detach().cpu().numpy().reshape(28, 28))
    
    fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolations[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"{i+1}")
    plt.show()

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform) 

model = autoencoderMLP4Layer()
model.load_state_dict(torch.load("MLP.8.pth"))
model.eval()

idx1, idx2 = get_two_ints()

img = train_set[idx1][0].view(1, -1)  
img2 = train_set[idx2][0].view(1, -1)

noise_factor = 0.2 # change this to change the intensity of the noise.
noisy_img = img + noise_factor*torch.rand(img.shape)


img = img.view(1,784)
img2 = img2.view(1,784)
noisy_img = noisy_img.view(1,784)

with torch.no_grad():
    # output_clean = model(img)
    output_clean = model(img)

with torch.no_grad():
    # output_noisy = model(img)
    output_noisy = model(noisy_img)


output_img_clean = output_clean.view(28, 28)
output_img_noisy = output_noisy.view(28,28)


f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(train_set.data[idx1], cmap='gray')   # original raw image
f.add_subplot(1, 2, 2)
plt.imshow(output_img_clean, cmap='gray')            # reconstructed image
plt.show()

f = plt.figure()
f.add_subplot(1, 3, 1)
plt.imshow(train_set.data[idx1], cmap='gray')   # original raw image
f.add_subplot(1, 3, 2)
plt.imshow(noisy_img.view(28,28), cmap = 'gray')
f.add_subplot(1, 3, 3)
plt.imshow(output_img_noisy, cmap='gray')            # reconstructed image
plt.show()

interpolate_between_imgs(model,img,img2)

