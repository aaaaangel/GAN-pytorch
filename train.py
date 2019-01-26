import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from visualization import generate_grid_result, show_train_loss, generate_gif

# Init weights by xavier.
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def train_GAN(generator, discriminator, train_loader, network_name, noise_dim=100, num_epochs=100, lr=0.0002, img_size=[28, 28]):
    # Init binary cross entropy loss function and Adam optimizer
    loss = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Use fixed noise to test generator after each epoch
    test_noise = torch.randn(5 * 5, noise_dim)
    if torch.cuda.is_available():
        test_noise = test_noise.cuda()

    d_epoch_losses = []
    g_epoch_losses = []
    # Train for epochs
    for epoch in range(num_epochs):
        time0 = time.time()
        d_batch_losses = []
        g_batch_losses = []
        for batch_real_images, _ in train_loader:
            cur_batch_size = batch_real_images.size(0)
            batch_real_images = batch_real_images.view(cur_batch_size, -1)
            batch_fake_noise = torch.randn(cur_batch_size, noise_dim)

            # Train discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(cur_batch_size)
            fake_labels = torch.zeros(cur_batch_size)

            if torch.cuda.is_available():  # use cuda
                batch_real_images = batch_real_images.cuda()
                batch_fake_noise = batch_fake_noise.cuda()
                real_labels = real_labels.cuda()
                fake_labels = fake_labels.cuda()

            batch_fake_images = generator(batch_fake_noise)

            d_real_prediction = discriminator(batch_real_images)
            d_real_loss = loss(d_real_prediction, real_labels)
            d_fake_prediction = discriminator(batch_fake_images)
            d_fake_loss = loss(d_fake_prediction, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            d_batch_losses.append(d_loss)

            # Train generator
            generator.zero_grad()
            batch_input_noise = torch.randn(cur_batch_size, noise_dim)
            expect_labels = torch.ones(cur_batch_size)
            if torch.cuda.is_available():  # use cuda
                batch_input_noise = batch_input_noise.cuda()
                expect_labels = expect_labels.cuda()
            batch_output_images = generator(batch_input_noise)
            g_prediction = discriminator(batch_output_images)
            g_loss = loss(g_prediction, expect_labels)
            g_loss.backward()
            g_optimizer.step()
            g_batch_losses.append(g_loss)

        d_epoch_losses.append(float(torch.Tensor(d_batch_losses).mean()))
        g_epoch_losses.append(float(torch.Tensor(g_batch_losses).mean()))

        results_dir = './results/'+network_name+'/'
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        test_path = results_dir + network_name + str(epoch + 1).zfill(3) + '.png'
        generate_grid_result(generator, test_path, input_noise=test_noise, epoch=epoch, img_size=img_size)

        time1 = time.time()

        print('[%d/%d]:\tloss_d: %.3f, loss_g: %.3f, time: %.3f' %
              (epoch, num_epochs, d_epoch_losses[-1], g_epoch_losses[-1], time1 - time0))

    graph_path = './results/' + network_name + '_loss.png'
    show_train_loss(g_epoch_losses, d_epoch_losses, graph_path)

    gif_path = './results/' + network_name + '_results.gif'
    generate_gif(results_dir, gif_path)



