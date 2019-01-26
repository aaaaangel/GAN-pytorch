# GAN-pytorch
Implement Generative Adversarial Networks (GAN)  and its variants using Pytorch step by step:

* GAN on MNIST
* DCGAN on MNIST
* ...



### Experiments and Results

---

#### GAN on MNIST

* Network structure

```Cmd
Generator(
  (fc1): Linear(in_features=100, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=1024, bias=True)
  (fc4): Linear(in_features=1024, out_features=784, bias=True)
)
Discriminator(
  (fc1): Linear(in_features=784, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=256, bias=True)
  (fc4): Linear(in_features=256, out_features=1, bias=True)
)
```

* Results

|           results after 100 epochs            |                   training process                    |                  training loss                  |
| :-------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------: |
| ![GAN_MNIST_100](./results/GAN_MNIST_100.png) | ![GAN_MNIST_results](./results/GAN_MNIST_results.gif) | ![GAN_MNIST_loss](./results/GAN_MNIST_loss.png) |

#### DCGAN on MNIST

- Network structure

```cmd
Generator(
  (deconv1): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1))
  (deconv1_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (deconv2): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (deconv2_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (deconv3): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (deconv3_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (deconv4): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (deconv4_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (deconv5): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
)
Discriminator(
  (conv1): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2_bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv3_bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv4_bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv5): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1))
)
```

- Results

|            results after 100 epochs             |                     training process                      |                    training loss                    |
| :---------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------: |
| ![DCGAN_MNIST020](./results/DCGAN_MNIST020.png) | ![DCGAN_MNIST_results](./results/DCGAN_MNIST_results.gif) | ![DCGAN_MNIST_loss](./results/DCGAN_MNIST_loss.png) |



### Reference

---

https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN



