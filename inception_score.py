from torchvision.models import inception_v3
import torch
from torch.autograd import Variable
import math
from sample import generate_noise

def inception_score(aG, n_images=500, batch_size=100, splits=10):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    n_batches = int(math.ceil(float(n_images)/float(batch_size)))

    def pred(x):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners = False).type(dtype)
            x = inception_model(x)
            x = torch.nn.functional.softmax(x, dim=1)

        return x

    preds = []

    for i in range(n_batches):
        noise, label = generate_noise(10, batch_size, 128)
        imgs = aG(noise, label)
        y = pred(imgs)
        preds.append(y)
    
    preds = torch.cat(preds, 0)

    split_scores = []
    
    for i in range(splits):
        part = preds[(i * (n_images // splits)): ((i+1) * (n_images // splits)), :]
        # py = np.mean(part, axis=0)
        # scores = []
        # for i in range(part.shape[0]):
        #     pyx = part[i, :]
        #     scores.append(entropy(pyx, py))
        # split_scores.append(np.exp(np.mean(scores)))
        kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        kl = torch.exp(kl)
        split_scores.append(kl.unsqueeze(0))
    
    split_scores = torch.cat(split_scores, 0)
    m_scores = torch.mean(split_scores).detach().cpu().numpy()
    m_std = torch.std(split_scores).detach().cpu().numpy()
    
    return m_scores, m_std