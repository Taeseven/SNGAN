import torch
from collections import OrderedDict
from torch.autograd import Variable
from layers import Linear, Conv2d, CategoricalConditionalBatchNorm2d

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = torch.nn.Sequential(
            OrderedDict(
                [
                    ('fc1', torch.nn.Linear(128, 512*4*4))
                ]
            )
        )

        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn1 = CategoricalConditionalBatchNorm2d(10, 256)
        self.conv2 = torch.nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn2 = CategoricalConditionalBatchNorm2d(10, 128)
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = CategoricalConditionalBatchNorm2d(10, 64)
        self.conv4 = torch.nn.ConvTranspose2d(64, 3, 3, 1, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, c):
        x = self.fc1(x)
        x = x.view(-1, 512, 4, 4)
        x = self.conv1(x)
        x = self.relu(self.bn1(x, c))
        x = self.conv2(x)
        x = self.relu(self.bn2(x, c))
        x = self.conv3(x)
        x = self.relu(self.bn3(x, c))
        x = self.conv4(x)
        img = self.tanh(x)
        
        return img

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = torch.nn.Sequential(
                OrderedDict(
                    [
                        ('conv1', Conv2d(3, 64, 3, 1, 1)),
                        ('leaky_relu1', torch.nn.LeakyReLU(0.1)),
                        ('conv2', Conv2d(64, 64, 4, 2, 1)),
                        ('leaky_relu2', torch.nn.LeakyReLU(0.1)),
                        ('conv3', Conv2d(64, 128, 3, 1, 1)),
                        ('leaky_relu3', torch.nn.LeakyReLU(0.1)),
                        ('conv4', Conv2d(128, 128, 4, 2, 1)),
                        ('leaky_relu4', torch.nn.LeakyReLU(0.1)),
                        ('conv5', Conv2d(128, 256, 3, 1, 1)),
                        ('leaky_relu5', torch.nn.LeakyReLU(0.1)),
                        ('conv6', Conv2d(256, 256, 4, 2, 1)),
                        ('leaky_relu6', torch.nn.LeakyReLU(0.1)),
                        ('conv7', Conv2d(256, 512, 3, 1, 1)),
                        ('leaky_relu7', torch.nn.LeakyReLU(0.1))
                    ]
                )
        )
        self.fc1 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ('fc', Linear(512*4*4, 1))
                    ]
                )
        )
        self.fc10 = torch.nn.Sequential(
                OrderedDict(
                    [
                        ('fc10', Linear(512*4*4, 10))
                    ]
                )
        )

    def forward(self, x):
        conv_res = self.conv(x)
        conv_res = conv_res.view(conv_res.size(0), -1)
        is_real = self.fc1(conv_res)
        label = self.fc10(conv_res)

        return is_real.squeeze(), label



if __name__ == '__main__':
    a = Discriminator()
    z = Variable(torch.randn(10, 3, 32, 32))
    out, _ = a(z)
    print(out)
    print(_.size())
