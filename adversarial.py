import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable
from linf_sgd import Linf_SGD
from loss import loss_nll

def attack_Linf_PGD(x, label, aD, steps, epsilon):
    aD.eval()
    x_adv = x.data.clone()
    x_adv = Variable(x_adv, requires_grad=True)
    optimizer = Linf_SGD([x_adv], lr=0.0078)
    for _ in range(steps):
        optimizer.zero_grad()
        aD.zero_grad()
        d_bin, d_multi = aD(x_adv)
        ones = d_bin.clone().fill_(1)
        ones = Variable(ones).cuda()
        loss = -loss_nll(d_bin, ones, d_multi, label)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = x_adv.data - x.data
        diff.clamp_(-epsilon, epsilon)
        x_adv.data.copy_((diff + x.data).clamp_(-1, 1))
    aD.train()
    aD.zero_grad()
    return x_adv

def attack_FGSM(x, label, aD, steps, epsilon):
    aD.eval()
    d_bin, d_multi = aD(x)
    ones = d_bin.clone().fill_(1)
    ones = Variable(ones).cuda()
    loss = -loss_nll(d_bin, ones, d_multi, label)
    g = grad(loss, [x], create_graph=True)[0]
    return x - 0.005 * torch.sign(g)

def attack_none(x, label, aD, steps, epsilon):
    return x
