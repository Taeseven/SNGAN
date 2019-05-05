import torch
import torch.nn.functional as F
from torch.nn import Parameter, init


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def max_singular_value(w_mat, u, power_iterations):

    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))
        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v



class Linear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_features), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)


    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_features, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)

            # w_bar = torch.div(w_mat, sigma)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
            # self.w_bar = w_bar.detach()
            # self.sigma = sigma.detach()
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_channels), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_channels, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.conv2d(input, w_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.num_embeddings), requires_grad=False))
        else:
            self.register_buffer("u", None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.embedding(
            input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class ConditionalBatchNorm2d(torch.nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = torch.nn.Embedding(num_classes, num_features)
        self.biases = torch.nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)

# class CategoricalConditionalBatchNorm2d(torch.nn.Module):

#     def __init__(self, num_features, num_categories, eps=1e-5, momentum=0.1, affine=False,
#                  track_running_stats=True):
#         super(CategoricalConditionalBatchNorm2d, self).__init__()
#         self.batch_norm = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
#         self.gamma_c = torch.nn.Embedding(num_categories, num_features)
#         self.beta_c = torch.nn.Embedding(num_categories, num_features)
#         torch.nn.init.constant_(self.batch_norm.running_var.data, 0)
#         torch.nn.init.constant_(self.gamma_c.weight.data, 1)
#         torch.nn.init.constant_(self.beta_c.weight.data, 0)

#     def forward(self, input, y):
#         ret = self.batch_norm(input)
#         gamma = self.gamma_c(y)
#         beta = self.beta_c(y)
#         gamma_b = gamma.unsqueeze(2).unsqueeze(3).expand_as(ret)
#         beta_b = beta.unsqueeze(2).unsqueeze(3).expand_as(ret)
#         return gamma_b*ret + beta_b