# EBGAN_pytorch

PyTorch implementation of [Energy-Based Generative Adversarial Networks] (https://arxiv.org/abs/1609.03126)

Some code derives from pytorch official example [dcgan] (https://github.com/pytorch/examples/tree/master/dcgan)

After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

##To do

Repelling regularizer

##Usage
```
  -h, --help            show this help message and exit
  --dataroot            path to dataset
  --workers             number of data loading workers
  --batchSize           input batch size
  --imageSize           the height / width of the input image to network             
  --nz                  size of the latent z vector
  --ngf 
  --ndf  
  --margin              margin of the energy loss
  --niter               number of epochs to train for
  --lr                  learning rate, default=0.0002
  --beta1               beta1 for adam. default=0.5
  --cuda                enables cuda
  --netG                path to netG (to continue training)
  --netD                path to netD (to continue training)
  --outf                folder to output images and model checkpoints
```
