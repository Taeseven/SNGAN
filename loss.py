import torch
import torch.nn.functional as F

# Binary loss
def loss_binary(bin_output, bin_label):
    return F.binary_cross_entropy_with_logits(bin_output, bin_label)

# KL loss
def loss_KL_d(dis_fake, dis_real):
    L1 = F.softplus(-dis_real).mean()
    L2 = F.softplus(dis_fake).mean()
    return L1, L2

def loss_KL_g(dis_fake):
    return F.softplus(-dis_fake).mean()

# Hinge loss
def loss_hinge_d(dis_fake, dis_real):
    L1 = F.relu(1.0 - dis_real).mean()
    L2 = F.relu(1.0 + dis_fake).mean()
    return L1, L2

def loss_hinge_g(dis_fake):
    return -dis_fake.mean()

# NLL loss
def loss_nll(bin_output, bin_label, multi_output, multi_label, lam=0.5):
    L1 = F.binary_cross_entropy_with_logits(bin_output, bin_label)
    L2 = F.cross_entropy(multi_output, multi_label)
    return lam * L1 + (1.0 - lam) * L2
