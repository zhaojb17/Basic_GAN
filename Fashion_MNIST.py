# Fashion_MNIST
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>. All Rights Reserved.
# This file is used for generating adversarial networks for Fashion MNIST.
# ***************************************************************

import time
import os
import argparse
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
parser.add_argument('--image_size', type=int, default=28, help="image size")
parser.add_argument('--epochs', type=int, default=200, help="epochs")
parser.add_argument('--noise_distribution', type=int, default=50, help="noise distribution")
parser.add_argument('--dis_lr', type=float, default=0.0001, help="discriminator learning rate")
parser.add_argument('--gen_lr', type=float, default=0.0002, help="generator learning rate")
parser.add_argument('--data_dir', type=str, default='E:\datasets', help="data dir")
parser.add_argument('--save_dir', type=str, default='./images/Fashion_MNIST/', help="data dir")
parser.add_argument('--log_dir', type=str, default='Fashion_MNIST_log.txt', help="log dir")
parser.add_argument('--sample_every', type=int, default=200, help="sample every steps")
args = parser.parse_args()

warnings.filterwarnings("ignore")

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.lrelu(self.conv_1(x))
        x = self.lrelu(self.conv_2(x))
        x = x.view(-1, 7 * 7 * 64)
        x = self.lrelu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(args.noise_distribution, 14 * 14)
        self.fc2 = nn.Linear(14 * 14, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1, 1, 28, 28)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img

def main():

    # Set the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_dir)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Generative Adversarial Networks")
    logger.info("Dataset:MNIST Epochs:{} Noise Distribution:{} Batch_size:{} Num_workers:{}".
                format(args.epochs, args.noise_distribution, args.batch_size, args.num_workers))
    logger.info("Discriminator learning rate:{} Generator learning rate:{} dataset dir:'{}' save dir:'{}' log dir:'{}'".
                format(args.dis_lr, args.gen_lr, args.data_dir, args.save_dir, args.log_dir))

    # Load the datasets
    tranformations = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=tranformations)
    test_dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=tranformations)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    net_d = discriminator().to(device)
    net_g = generator().to(device)

    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=args.dis_lr)
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=args.gen_lr)
    criterion = nn.BCELoss().to(device)

    true_labels = torch.tensor(torch.ones(args.batch_size)).to(device)
    fake_lable = torch.tensor(torch.zeros(args.batch_size)).to(device)

    for epoch in range(args.epochs):
        start_time = time.time()
        for i, (images, _) in enumerate(train_loader):

            real_images = images.to(device)
            noises = np.random.randn(args.batch_size, args.noise_distribution)

            dis_labels = net_d(real_images)
            optimizer_d.zero_grad()
            loss_d_real = criterion(dis_labels, true_labels)
            loss_d_real.backward()

            fake_in = torch.Tensor(noises).to(device)
            fake_images = net_g(fake_in)

            dis_fake_labels = net_d(fake_images)
            loss_d_fake = criterion(dis_fake_labels, fake_lable)
            loss_d_fake.backward(retain_graph=True)
            optimizer_d.step()

            fake_images = net_g(fake_in)
            optimizer_g.zero_grad()
            d_fake_labels = net_d(fake_images)
            g_loss = criterion(d_fake_labels, true_labels)
            g_loss.backward()
            optimizer_g.step()

            if i % args.sample_every == 0:
                end_time = time.time()
                plt.clf()
                logger.info("Epoch:{} time:{:.2f}s step:{} dis_real_loss:{:.2f} dis_fake_loss:{:.2f} gen_loss:{:.2f}".
                            format(epoch, end_time - start_time, i, loss_d_real.item(), loss_d_fake.item(), g_loss.item()))
                plt.imshow(merge(fake_images.to('cpu').detach().numpy().transpose((0, 2, 3, 1)), [8, 8]))
                plt.text(-10.0, -5.0, 'Epoch:{} step:{} D accuracy={:.2f} (0.5 for D to converge)'.
                         format(epoch, i, (dis_labels.mean() + 1 - d_fake_labels.mean()) / 2), fontdict={'size': 10})
                plt.draw()
                plt.savefig(args.save_dir + str(epoch) + ' ' + str(i // args.sample_every) + '.png')
                start_time = time.time()

if __name__ == '__main__':
    main()