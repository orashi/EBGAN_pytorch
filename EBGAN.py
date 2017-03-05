from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import math


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# folder dataset
dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.convT1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        # state size. (ngf*8) x 4 x 4

        self.convT2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        # state size. (ngf*4) x 8 x 8

        self.convT3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        # state size. (ngf*2) x 16 x 16

        self.convT4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x 32 x 32

        self.convT5 = nn.ConvTranspose2d(ngf, nc, 5, 3, 1, bias=False)
        # state size. (nc) x 96 x 96

    def forward(self, x):

        out = F.relu(self.bn1(self.convT1(x)), True)
        out = F.relu(self.bn2(self.convT2(out)), True)
        out = F.relu(self.bn3(self.convT3(out)), True)
        out = F.relu(self.bn4(self.convT4(out)), True)
        out = F.tanh(self.convT5(out))

        return out

netG = _netG()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        # input is (nc) x 96 x 96 (for anime image)
        self.enc_conv1 = nn.Conv2d(nc, ndf, 5, 3, 1, bias=False)
        # state size. (ndf) x 32 x 32

        self.enc_conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # 3 stride for 96
        self.enc_bn2 = nn.BatchNorm2d(ndf * 2)
        # state size. (ndf*2) x 16 x 16

        self.enc_conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.enc_bn3 = nn.BatchNorm2d(ndf * 4)
        # state size. (ndf*4) x 8 x 8

        self.enc_conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.enc_bn4 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf*8) x 4 x 4

        self.dec_conv4 = nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False)
        self.dec_bn4 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf*4) x 8 x 8

        self.dec_conv3 = nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False)
        self.dec_bn3 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf*2) x 16 x 16

        self.dec_conv2 = nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False)
        self.dec_bn2 = nn.BatchNorm2d(ndf * 8)
        # state size. (ndf) x 32 x 32

        self.dec_conv1 = nn.ConvTranspose2d(ndf, 3, 5, 3, 0, bias=False)
        # state size. 3 x 96 x 96
        ''' stride improvable '''

    def forward(self, x):
        out = F.leaky_relu(self.enc_conv1(x), 0.2, True)
        out = F.leaky_relu(self.enc_bn2(self.enc_conv2(out)), 0.2, True)
        out = F.leaky_relu(self.enc_bn3(self.enc_conv3(out)), 0.2, True)
        out = F.leaky_relu(self.enc_bn4(self.enc_conv4(out)), 0.2, True)

        ''' question: fc layer???? '''

        out = F.leaky_relu(self.dec_bn4(self.dec_conv4(out)), 0.2, True)
        out = F.leaky_relu(self.dec_bn3(self.dec_conv3(out)), 0.2, True)
        out = F.leaky_relu(self.dec_bn2(self.dec_conv2(out)), 0.2, True)
        out = self.dec_conv1(out)

        return out

netD = _netD()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()   # criterion over here!!!

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        # prepare real
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        # train with real
        output = netD(input)
        errD_real = criterion(output, label)  # score on real
        errD_real.backward()  # backward on score on real
        D_x = output.data.mean()  # score fore supervision

        # generate fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise)
        label.data.fill_(fake_label)

        # train with fake
        output = netD(fake.detach())
        errD_fake = criterion(output, label)  # score on fake
        errD_fake.backward()  # backward on score on fake
        D_G_z1 = output.data.mean()  # score fore supervision
        errD = errD_real + errD_fake  # score fore supervision

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()

        # reuse generated fake samples
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()

        optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
