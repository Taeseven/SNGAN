import numpy as np
import torch 
from torch.autograd import Variable


def generate_noise(n_classes, batch_size, n_z):
    cuda = True if torch.cuda.is_available() else False

    label = np.random.randint(0,n_classes,batch_size)
    noise = np.random.normal(0,1,(batch_size,n_z))
    label_onehot = np.zeros((batch_size,n_classes))
    label_onehot[np.arange(batch_size), label] = 1
    noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
    noise = noise.astype(np.float32)
    noise = torch.from_numpy(noise)
    fake_label = torch.from_numpy(label)
    if cuda:
        noise = Variable(noise).cuda()
        fake_label = Variable(fake_label).cuda()

    return noise, fake_label
