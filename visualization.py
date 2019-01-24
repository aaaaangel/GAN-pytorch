import imageio
import os
import matplotlib.pyplot as plt

import torch


def generate_grid_result(generator, save_path, input_noise=None, noise_dim=100, grid_side=5, img_size=(28,28), epoch=None):
    if input_noise is None:
        input_noise = torch.randn(grid_side*grid_side, noise_dim)
        if torch.cuda.is_available():
            input_noise = input_noise.cuda()
    output_images = generator(input_noise)

    fig, ax = plt.subplots(grid_side, grid_side)

    for i in range(grid_side):
        for j in range(grid_side):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].imshow(output_images[i*grid_side+j, :].cpu().data.view(img_size, img_size).numpy(), cmap='gray')

    if epoch is not None:
        label = 'Epoch '+str(epoch)
        fig.text(0.5, 0.04, label, ha='center')

    plt.savefig(save_path)
    plt.close()


def show_train_loss(g_loss, d_loss, save_path):
    x = range(len(g_loss))
    plt.plot(x, g_loss, label='generator_loss')
    plt.plot(x, d_loss, label='discriminator_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_gif(imgs_dir, save_path):
    image_names = os.listdir(imgs_dir)
    image_names.sort()
    images = []
    for img_name in image_names:
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            images.append(imageio.imread(imgs_dir+img_name))
    imageio.mimsave(save_path, images, fps=5)


