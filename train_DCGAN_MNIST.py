import torch
from torchvision import datasets, transforms

from models import DCGAN_MNIST
from train import weights_init, train_GAN


if __name__ == '__main__':
    # training settings
    batch_size = 128
    lr = 0.0002
    num_epochs = 100
    noise_dim = 100
    img_size = 64

    # Init network
    generator = DCGAN_MNIST.Generator(input_size=noise_dim)
    discriminator = DCGAN_MNIST.Discriminator()
    print(generator)
    print(discriminator)
    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    weights_init(generator)
    weights_init(discriminator)

    # Load data
    transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)

    train_GAN(generator, discriminator, train_loader, 'DCGAN_MNIST', num_epochs=20, img_size=[64,64])
