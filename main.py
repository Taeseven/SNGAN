import time
import os
import sys
import argparse
import numpy as np 
import torch
import torchvision
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sample import generate_noise
from inception_score import inception_score
from loss import *
from plot import plot_grad_flow, plot_sample
from gradient_penalty import calc_gradient_penalty
from adversarial import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='ResNet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoch', type=int, default=999)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_z', type=int, default=128)
parser.add_argument('--gen_train', type=int, default=5)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--gradient_dir', type=str, default='./output/gradient/')
parser.add_argument('--image_dir', type=str, default='./output/image/')
parser.add_argument('--adv', type=str, default='1')
parser.add_argument('--adv_steps', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.0625)
opt = parser.parse_args()

CUDA = True if torch.cuda.is_available() else False
if CUDA:
    torch.cuda.set_device(opt.ngpu)

def check_dir():
    if not os.path.exists('./data/'):
        os.makedirs('./data/' )
    if not os.path.exists(opt.gradient_dir):
        os.makedirs(opt.gradient_dir)
    if not os.path.exists(opt.image_dir):
        os.makedirs(opt.image_dir)

def get_loss():
    if opt.loss == 'kl':
        return loss_KL_g, loss_KL_d
    if opt.loss == 'binary':
        return loss_binary, loss_binary
    if opt.loss == 'hinge':
        return loss_hinge_g, loss_hinge_d
    if opt.loss == 'nll':
        return loss_nll, loss_nll

def load_models():
    if opt.model == 'sn':
        sys.path.append('./models')
        from models_sn import Generator, Discriminator
        aG = Generator()
        aD = Discriminator()
    if opt.model == 'wgan-gp':
        # from gradient_penalty import calc_gradient_penalty
        return None
    if opt.model == 'ResNet':
        sys.path.append('./models')
        from models_resnet import Generator, Discriminator

        aG = Generator(num_classes=opt.n_classes)
        aD = Discriminator(num_classes=opt.n_classes)
    return aG, aD

def make_dataset():
    def noise(x):
        return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
    
    if opt.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(
                    brightness=0.1*torch.rand(1).item(),
                    contrast=0.1*torch.rand(1).item(),
                    saturation=0.1*torch.rand(1).item(),
                    hue=0.1*torch.rand(1).item()),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(noise)
            ])

        transform_test = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        train_set = torchvision.datasets.CIFAR10(root='./data/', 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            download=False, 
                                            transform=transform_test)

    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")

    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=opt.batch_size, 
                                            shuffle=True, 
                                            num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                            batch_size=opt.batch_size, 
                                            shuffle=False, 
                                            num_workers=8)

    return train_loader, test_loader

def get_adv():
    if opt.adv == 'PGD':
        return attack_Linf_PGD
    if opt.adv == 'FGSM':
        return attack_FGSM
    else:
        return attack_none

def eval_model(aD, test_loader):
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            X_test_batch, Y_test_batch = Variable(X_test_batch), Variable(Y_test_batch)
            if CUDA:
                X_test_batch = X_test_batch.cuda()
                Y_test_batch = Y_test_batch.cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1]
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(opt.batch_size))*100.0
            test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)

    return accuracy_test 

def main():
    check_dir()
    aG, aD = load_models()
    train_loader, test_loader = make_dataset()
    loss_g, loss_d = get_loss()
    loss_c = torch.nn.CrossEntropyLoss()
    adv_func = get_adv()

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=1e-3, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=2e-4, betas=(0,0.9))
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_g, step_size=200, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_d, step_size=200, gamma=0.5)

    # generate noise
    np.random.seed(352)
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0,1, (100, opt.n_z))
    label_onehot = np.zeros((100, opt.n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :opt.n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)
    save_noise = torch.from_numpy(noise)
    label = torch.from_numpy(label)

    if CUDA:
        aD.cuda()
        aG.cuda()
        save_noise = save_noise.cuda()
        label = label.cuda()


    for epoch in range(opt.num_epoch):
        aD.train()
        aG.train()
        start_time = time.time()

        for group in optimizer_g.param_groups:
            for p in group['params']:
                state = optimizer_g.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer_d.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        print('---------------------------------')
        print('Training at Epoch ', epoch)

        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        acc1 = []

        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
            if (Y_train_batch.shape[0] < opt.batch_size):
                continue     

            # train generator
            if (batch_idx % opt.gen_train == 0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()
                noise, fake_labels = generate_noise(opt.n_classes, opt.batch_size, opt.n_z)
            
                fake_data = aG(noise, fake_labels)
                gen_source, gen_class = aD(fake_data, fake_labels)

                gen_class = loss_c(gen_class, fake_labels)
                gen_source = loss_g(gen_source)

                gen_cost = gen_source + gen_class
                gen_cost.backward()

                if((batch_idx%150)==0):
                    fig = plot_grad_flow(aG.named_parameters())
                    png_name = 'aG_' + str(epoch).zfill(3) + '_' + str(batch_idx).zfill(3) + '.png'
                    plt.savefig('%s/%s' % (opt.gradient_dir, png_name), bbox_inches='tight')
                    plt.close(fig)

                optimizer_g.step()

            # train discriminator
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()  
            real_data, real_class = Variable(X_train_batch), Variable(Y_train_batch)
            if CUDA:
                real_data = real_data.cuda()
                real_class = real_class.cuda()

            x_adv = adv_func(real_data, real_class, aD, opt.adv_steps, opt.epsilon)

            disc_real_source, disc_real_class = aD(x_adv, real_class)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_class.data).sum() ) /float(opt.batch_size))*100.0

            noise, fake_label = generate_noise(opt.n_classes, opt.batch_size, opt.n_z)
            with torch.no_grad():
                fake_data = aG(noise, fake_label)

            disc_fake_source, disc_fake_class = aD(fake_data, fake_label)
            
            disc_fake_source, disc_real_source = loss_d(disc_fake_source, disc_real_source)
            disc_real_class = loss_c(disc_real_class, real_class)
            disc_fake_class = loss_c(disc_fake_class, fake_label)

            disc_cost =  disc_fake_source + disc_real_source + disc_fake_class + disc_real_class

            if opt.model == 'wgan-gp':
                gradient_penalty = calc_gradient_penalty(aD, x_adv,fake_data, opt.batch_size)
                disc_cost += gradient_penalty

            disc_cost.backward()

            if((batch_idx%150)==0):
                fig = plot_grad_flow(aG.named_parameters())
                png_name = 'aD_' + str(epoch).zfill(3) + '_' + str(batch_idx).zfill(3) + '.png'
                plt.savefig('%s/%s' % (opt.gradient_dir, png_name), bbox_inches='tight')
                plt.close(fig)

            optimizer_d.step()

            loss1.append(disc_fake_source.item())
            loss2.append(disc_real_source.item())
            loss3.append(disc_real_class.item())
            loss4.append(disc_fake_class.item())
            # loss3.append(0)
            # loss4.append(0)
            acc1.append(accuracy)

            if((batch_idx%50)==0):
                print(batch_idx, "%.2f" % np.mean(loss1), 
                                "%.2f" % np.mean(loss2), 
                                "%.2f" % np.mean(loss3), 
                                "%.2f" % np.mean(loss4), 
                                "%.2f" % np.mean(acc1))

        print('Epoch ', epoch, 'time cost: ',time.time()-start_time)
        scheduler_D.step()
        scheduler_G.step()

        aD.eval()
        aG.eval()
        with torch.no_grad():
            accuracy_class = eval_model(aD, test_loader)
            print('Epoch ', epoch, 'Class-testing accu: ', accuracy_class) 

        if((epoch%10)==0):
            with torch.no_grad():           
                samples = aG(save_noise, label)
                samples = samples.data.cpu().numpy()
                samples += 1.0
                samples /= 2.0
                samples = samples.transpose(0,2,3,1)

            fig = plot_sample(samples)
            png_name = str(epoch).zfill(3) + '.png'
            plt.savefig('%s/%s' % (opt.image_dir, png_name), bbox_inches='tight')
            plt.close(fig)

            scores, std = inception_score(aG)
            print('Epoch ', epoch, 'Inception Scores: ', scores, std)

            torch.save(aG,'tempG.model')
            torch.save(aD,'tempD.model')
            
if __name__ == '__main__':
    main()
    
    