import torch
import torch.nn.functional as F
from resblock import ResBlock_G, ResBlock_D, ResBlock_D_opt
from layers import Linear, Embedding

class Generator(torch.nn.Module):
    def __init__(self, num_features=64, n_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0):
        super(Generator, self).__init__()

        self.num_features = num_features
        self.n_z = n_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes

        self.fc = torch.nn.Linear(self.n_z, 16 * self.num_features * self.bottom_width ** 2)
        self.block1 = ResBlock_G(self.num_features  * 16, self.num_features * 8,
                                activation=self.activation, upsample=True, num_classes=self.num_classes)
        self.block2 = ResBlock_G(self.num_features  * 8, self.num_features * 4,
                                activation=self.activation, upsample=True, num_classes=self.num_classes)
        self.block3 = ResBlock_G(self.num_features  * 4, self.num_features * 2,
                                activation=self.activation, upsample=True, num_classes=self.num_classes)
        self.block4 = ResBlock_G(self.num_features  * 2, self.num_features,
                                activation=self.activation, upsample=True, num_classes=self.num_classes)
        self.bn = torch.nn.BatchNorm2d(self.num_features)
        self.conv = torch.nn.Conv2d(self.num_features, 3, 1, 1)
        self.tanh = torch.nn.Tanh()
        self._initialize()
    
    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x, y=None):
        x = self.fc(x)
        x = x.view(-1, 16 * self.num_features, self.bottom_width, self.bottom_width)
        x = self.block1(x, y)
        x = self.block2(x, y)
        x = self.block3(x, y)
        x = self.block4(x, y)
        x = self.activation(self.bn(x))
        x = self.tanh(self.conv(x))
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(Discriminator, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = ResBlock_D_opt(3, self.num_features)
        self.block2 = ResBlock_D(self.num_features, self.num_features * 2,
                                activation=self.activation, downsample=True)
        self.block3 = ResBlock_D(self.num_features * 2, self.num_features * 4,
                                activation=self.activation, downsample=True)
        self.block4 = ResBlock_D(self.num_features * 4, self.num_features * 8,
                                activation=self.activation, downsample=True)
        self.block5 = ResBlock_D(self.num_features * 8, self.num_features * 16,
                                activation=self.activation, downsample=True)
        self.fc1 = Linear(self.num_features * 16, 1)      
        if self.num_classes > 0:
            self.emb1 = Embedding(self.num_classes, num_features * 16)
            self.fc10 = Linear(self.num_features * 16, self.num_classes)
        self._initialize()
        
    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        if self.num_classes > 0:
            torch.nn.init.xavier_uniform_(self.fc10.weight)
        
    def forward(self, x, y=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2,3))
        source = self.fc1(x)
        if y is not None:
            source += torch.sum(self.emb1(y) * x, dim=1, keepdim=True)
        if self.num_classes > 0:
            label = self.fc10(x)
            return source, label
        else:
            return source

if __name__ == '__main__':
    a = Generator(num_classes=10)